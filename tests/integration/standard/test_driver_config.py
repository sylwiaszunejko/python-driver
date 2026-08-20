# Copyright 2026 ScyllaDB, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import time
import unittest

from cassandra.driver_config import (DRIVER_CONFIG_OPTION, DRIVER_CONFIG_SCHEMA_VERSION,
                                     SESSION_ID_OPTION)
from tests.integration import (TestCluster, get_client_options, use_single_node, remove_cluster,
                               xfail_scylla_version_lt)


def setup_module():
    # A single node keeps the clients table, which lists only the connections
    # made to the node serving the query, complete: every connection of the
    # cluster under test is made to that one node.
    use_single_node()


def teardown_module():
    remove_cluster()


CONNECTION_WAIT_TIMEOUT = 30


def _expected_connection_count(session):
    """
    Number of connections the driver believes it has open: the control
    connection plus every live pool connection, one per shard on a shard aware
    cluster. This is what "every connection of this cluster" has to be counted
    against; asserting over whichever connections happen to be listed says
    nothing about the ones that are not.
    """
    return 1 + sum(state['open_count'] for state in session.get_pool_state().values())


def _settled_connection_count(session, timeout=CONNECTION_WAIT_TIMEOUT):
    """
    Same count, taken once the pools have stopped filling.

    ``Cluster.connect(wait_for_all_pools=True)`` waits only for each pool's
    first connection. On a shard aware cluster HostConnection opens that one,
    learns the shard count from it, and then submits a connection per remaining
    shard to the session executor, returning before any of them are up. Counting
    straight after connect would therefore set the bar at one connection per
    host and let the assertions pass without ever looking at the shard
    connections.

    A connection that fails to open for good would keep the count below the
    shard count forever, so this waits for the count to stop moving rather than
    for a particular value, and returns whatever the driver ended up with.
    """
    deadline = time.time() + timeout
    # Two unchanged reads in a row, so that a gap between two shard connections
    # coming up is not mistaken for the pools having settled.
    settled_reads, previous = 0, None
    while True:
        count = _expected_connection_count(session)
        settled_reads = settled_reads + 1 if count == previous else 0
        previous = count
        if settled_reads >= 2 or time.time() >= deadline:
            return count
        time.sleep(0.5)


def _wait_for_connections(session, session_id, count, timeout=CONNECTION_WAIT_TIMEOUT):
    """
    Polls the clients table until at least ``count`` connections report
    ``session_id``, and returns the client options of the ones that do.

    ``Cluster.connect(wait_for_all_pools=True)`` waits for the pools to be
    created, but a connection shows up here only once the server has registered
    it, so the rows arrive later than the connections do.

    Returns a short list if the timeout expires first rather than asserting, so
    that the count stays the caller's claim to make: an "absent everywhere" or a
    "reported exactly once" holds trivially over a list that is short only
    because the rows had not appeared yet.
    """
    deadline = time.time() + timeout
    while True:
        options = [o for o in get_client_options(session) if o.get(SESSION_ID_OPTION) == session_id]
        if len(options) >= count or time.time() >= deadline:
            return options
        time.sleep(0.5)


def _assert_listed(options, count, session_id, timeout=CONNECTION_WAIT_TIMEOUT):
    assert len(options) >= count, \
        "only %d of %d connections with SESSION_ID %s were listed within %ss" % (
            len(options), count, session_id, timeout)


@xfail_scylla_version_lt(reason='scylladb/scylla-enterprise#5467 - system.client_options is not yet supported',
                         scylla_version="2026.1.0")
class DriverConfigReportingTest(unittest.TestCase):
    def test_every_connection_reports_the_session_id(self):
        """
        The session id is what correlates a cluster's connections with each
        other in the clients table, so all of them must report the one the
        cluster exposes -- not merely some of them.
        """
        cluster = TestCluster()
        try:
            session = cluster.connect(wait_for_all_pools=True)
            session_id = str(cluster.session_id)
            expected = _settled_connection_count(session)

            options = _wait_for_connections(session, session_id, count=expected)

            # The rows were selected by session id, so the count is the claim:
            # as many connections carry it as the cluster has open. More is
            # fine, a connection that has already been closed lingers in the
            # table; fewer means one of them went out without the id.
            _assert_listed(options, expected, session_id)
        finally:
            cluster.shutdown()

    def test_distinct_clusters_report_distinct_session_ids(self):
        cluster = TestCluster()
        other_cluster = TestCluster()
        try:
            session = cluster.connect(wait_for_all_pools=True)
            other_cluster.connect(wait_for_all_pools=True)

            session_id = str(cluster.session_id)
            other_session_id = str(other_cluster.session_id)
            assert session_id != other_session_id

            # Both clusters reach the same node, so either session can read the
            # rows of both. Each id is waited for rather than read straight out
            # of one snapshot: connect() returns once the pools exist on the
            # client, while the rows appear only once the server has registered
            # the connections, so whichever cluster registered last would
            # intermittently be missing.
            #
            # One row per cluster is the whole claim here -- that the two ids
            # both reach the server and differ. Counting every connection of a
            # cluster is what test_every_connection_reports_the_session_id is
            # for.
            for wanted in (session_id, other_session_id):
                _assert_listed(_wait_for_connections(session, wanted, count=1), 1, wanted)
        finally:
            other_cluster.shutdown()
            cluster.shutdown()

    def test_only_the_control_connection_reports_the_driver_config(self):
        """
        The configuration is the same for every connection of a cluster, so it is
        reported once, by the control connection, to keep the other STARTUP
        frames small.
        """
        cluster = TestCluster()
        try:
            session = cluster.connect(wait_for_all_pools=True)
            session_id = str(cluster.session_id)

            # Every connection of the cluster, not merely two of them. Two rows
            # can both be pool connections, and the control connection is the
            # only one that ever reports, so its row missing from the snapshot
            # would empty `reports` and fail this test with nothing broken.
            expected = _settled_connection_count(session)
            options = _wait_for_connections(session, session_id, count=expected)
            _assert_listed(options, expected, session_id)

            reports = [o[DRIVER_CONFIG_OPTION] for o in options if DRIVER_CONFIG_OPTION in o]

            # Rows can outlive the connections they describe, so the table may
            # list more than the cluster has open. Those extra rows are not
            # harmless bystanders: a control connection re-established during
            # this test leaves a closed row carrying this same session id and a
            # DRIVER_CONFIG of its own, so a second report does not by itself
            # mean a pool connection produced one.
            #
            # The exact count is kept regardless, because relaxing it would let
            # through the regression this test exists for, and a reconnect
            # against a healthy single node within the seconds it runs is not
            # expected. The message says so, so that a failure here can be told
            # apart from the regression.
            assert len(reports) == 1, \
                ("expected exactly one connection to report %s, got %d. If the control "
                 "connection reconnected during this test, the closed one may still be "
                 "listed with a report of its own." % (DRIVER_CONFIG_OPTION, len(reports)))
            assert json.loads(reports[0]) == {'version': DRIVER_CONFIG_SCHEMA_VERSION}
        finally:
            cluster.shutdown()

    def test_driver_config_is_not_reported_when_disabled(self):
        """
        With reporting disabled not even the control connection reports
        DRIVER_CONFIG, while the session id, which the setting is documented not
        to affect, is still reported by every connection.
        """
        cluster = TestCluster(driver_config_reporting_enabled=False)
        try:
            session = cluster.connect(wait_for_all_pools=True)
            session_id = str(cluster.session_id)

            # Every connection of the cluster, not merely two of them. Two rows
            # can both be pool connections, which never report whatever this
            # setting says, so the absence below would hold over them even with
            # the control connection reporting away: the one case this test
            # exists to catch.
            expected = _settled_connection_count(session)
            options = _wait_for_connections(session, session_id, count=expected)
            _assert_listed(options, expected, session_id)

            assert all(DRIVER_CONFIG_OPTION not in o for o in options)
        finally:
            cluster.shutdown()

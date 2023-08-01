#! /bin/bash -e

sudo apt-get install gcc python3-dev libev4 libev-dev

aio_max_nr_recommended_value=1048576
aio_max_nr=$(cat /proc/sys/fs/aio-max-nr)
echo "The current aio-max-nr value is $aio_max_nr"
if (( aio_max_nr !=  aio_max_nr_recommended_value  )); then
   sudo sh -c  "echo 'fs.aio-max-nr = $aio_max_nr_recommended_value' >> /etc/sysctl.conf"
   sudo sysctl -p /etc/sysctl.conf
   echo "The aio-max-nr was changed from $aio_max_nr to $(cat /proc/sys/fs/aio-max-nr)"
   if (( $(cat /proc/sys/fs/aio-max-nr) !=  aio_max_nr_recommended_value  )); then
     echo "The aio-max-nr value was not changed to $aio_max_nr_recommended_value"
     exit 1
   fi
fi

SCYLLA_RELEASE='release:5.1'

python3 -m venv .test-venv
source .test-venv/bin/activate
pip install -U pip wheel setuptools

# install driver wheel
pip install --ignore-installed -r test-requirements.txt pytest
pip install -e .

# download awscli
pip install awscli

# install scylla-ccm
pip install https://github.com/scylladb/scylla-ccm/archive/master.zip

# download version

ccm create scylla-driver-temp -n 1 --scylla --version ${SCYLLA_RELEASE}
ccm remove

# run test

echo "export SCYLLA_VERSION=${SCYLLA_RELEASE}"
echo "PROTOCOL_VERSION=4 EVENT_LOOP_MANAGER=asyncio pytest --import-mode append tests/integration/standard/"
export SCYLLA_VERSION=${SCYLLA_RELEASE}
export MAPPED_SCYLLA_VERSION=3.11.4
PROTOCOL_VERSION=4 EVENT_LOOP_MANAGER=libev pytest -rf --import-mode append $*

# download version

export SCYLLA_UNSTABLE='unstable/master:2023-07-31T05:54:06Z'
export SCYLLA_VERSION=${SCYLLA_UNSTABLE}

ccm create scylla-driver-tablets-temp -n 3 --scylla --version ${SCYLLA_UNSTABLE}
ccm remove

# run tablet tests

PROTOCOL_VERSION=4 EVENT_LOOP_MANAGER=libev pytest -rf --import-mode append tests/integration/standard/test_tablets.py

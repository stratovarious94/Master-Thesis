bootstrap:
    sudo apt-get install python3-pip
	sudo pip3 install virtualenv
	virtualenv thesis2019
	thesis2019/bin/pip install --upgrade pip
	thesis2019/bin/pip install --upgrade setuptools
	thesis2019/bin/pip install --upgrade wheel
	thesis2019/bin/pip install -e .
	thesis2019/bin/pip install -r requirements.txt
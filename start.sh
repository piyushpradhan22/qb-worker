#!/bin/bash
echo "Starting Service from bash!!!"

apt-cache madison ${package}
python3 write_config.py
${package} --webui-port=3333 -d
python3 set_completion_script.py
sleep 2
start-stop-daemon --stop --name ${package}
sleep 2
echo "Final start at 7860!!!"
${package} --webui-port=7860

#!/usr/bin/bash
rsync -r goergmat@arcade.informatik.hu-berlin.de:/home/goergmat/bainf/src/output/ /home/matthias/simexpal-launch/output/
python eval.py

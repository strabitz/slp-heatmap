# slp-heatmap

This is a program inspired by the SLP Enforcer coordinate visualizer
(https://github.com/altf4/enforcer) which creates a heatmap visualization of
a user's stick inputs. This is a work in progress. Ideally this would be one
python script, but slippi-py is giving me weird/wrong stick inputs and its
successor peppi-py isn't working on my laptop. In the future I'll get a
website or standalone app up as well.

## Installation
Run `yarn install` and `pip install -r requirements.txt` to install the
necessary requirements for node and python.

Install `ffmpeg` in order to generate videos.

Install `ts-node` with `npm install ts-node`.

## Running the program
Generate the coordinate jsons with `ts-node coords/src/index.ts <slp_file> <output_path>`

Then generate the heatmap with `python heatmap/heatmap.py <coordinate_json> --mode <image or video> --output <output_path>`

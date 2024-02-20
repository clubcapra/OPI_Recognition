# ERICards object finder
For more information on ERICards, [see this](https://www.ericards.net).

## Setup
### Install requirements

Install tesseract-ocr [see this](https://pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/)

Then setup a venv:
```bash
python3 -m venv venv
. venv/bin/activate
python -m pip install pip --upgrade
pip install -r requirements.txt
```

## Usage
### To add more samples
1. Run the program at least once (it will create the required folders).
2. Download images and put them in the `samples` forlder.
3. By convention, rename the image to a number not present in the `converted` folder.

#### Currently tested image types:
| Format | Supported |
| ------ | --------- |
| .png   | ✔️         |
| .jpg   | ✔️         |
| .jpeg  | ✔️         |
| .webp  | ✔️         |
| .avif  | ❌        |


### Benchmarking

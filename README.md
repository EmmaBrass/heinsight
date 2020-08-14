# HeinSight - Liquid-Level

*HeinSight* is a generalizable computer-vision based system capable of monitoring and controlling liquid-level across a variety of chemistry applications that require continous stirring. It is comprised of three main components:

1. a pump(s) 
2. a webcam
3. a series of custom Python scripts

The Hydrogen-1 release is a prototype, designed to be inexpensive and easy to use. It comes complete with a GUI that
enables users to set-up and run the system without having to interact with code. This release is associated with the
[Automated Liquid-Level Monitoring and Control using Computer Vision](https://doi.org/10.26434/chemrxiv.12798143.v1)

## Getting Started

System requirements are briefly detailed below. For a complete guide to getting started and using the GUI, see `gui_user_docs.pdf`. 
Documentation related to the code can be found in docstrings and comments included in each script. 

### Hardware Requirements

All applications are currently programmed to run using one or two [NE-9000 peristaltic pumps](http://www.syringepump.com/peristaltic.php) and a [Logitech C922x Pro Stream Webcam](https://www.logitech.com/en-ca/product/c922-pro-stream-webcam).  

Other hardware may be used but the code will need to be modified accordingly.

### Software Requirements

- Development and testing has only been done on Windows 7 Enterprise and Winodws 10 Enterprise
- [PyCharm](https://www.jetbrains.com/pycharm/) (or other IDE)
- [Python3](https://www.python.org/)
    - We have tested that the code is compatible with Python versions 3.6.5 - 3.7.3
- [Logitech Camera Settings](https://support.logi.com/hc/en-us/articles/360024695174-Downloads-C920s-HD-Pro-Webcam) (or other webcam control software)
- HeinSight Liquid-Level scripts (found in this repo)
    - See the `requirements.txt`file for a complete list of necessary dependencies.

## How to use

See the user documentation detailed in `gui_user_docs.pdf`.

## Future Developement

*HeinSight* is an ongoing project. Development is underway to automate other common laboratory tasks based on computer vision. This repo will remain static. Updated releases as well as any publications or projects related to the system will be posted to
the HeinSight website.  

## Authors
- **Veronica Lai** - Coding, data visualization
- **Tara Zepel** - Coding, data visualization and analysis
- **Lars Yunker** - Initial work, tech support

## Acknowledgements 
- **Jason Hein** - Chemistry support
- **Josh Derasp** - Chemistry support
- **Sean Clark** - Tech support
- **Brad Buran** - Peristaltic pump control is based off of Buran's [new era pump interface](https://new-era-pump.readthedocs.io/en/latest/) 

## License
The code is licensed under the GPL v3.

from flask import url_for
from datetime import date
from markupsafe import Markup

import os


def sayHello():
	today = str(date.today())
	markup = Markup(f'<code>{today}</code>')
	return markup


def makeMessage(message):
	return message


def makePlots():
	# Use url_for to create a web path for an asset
	img_url = url_for('static', filename='outputs/images/img.png')
	markup = Markup(f'<img src="{img_url}">')
	return markup


def makeText():
	# Use os.path for inserting content
	full_filename = os.path.join(os.getcwd() + '/static/outputs/markdown', 'textfile.md')
	text = open(full_filename, 'r')
	return text

def run_hello():
    with open("hello.txt", "w") as f:
        f.write("Hello World")
    return "hello.txt"

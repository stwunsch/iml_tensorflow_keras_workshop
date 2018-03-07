#!/bin/bash

pandoc -t beamer -s -fmarkdown-implicit_figures --template=template.beamer slides.md -o slides.pdf

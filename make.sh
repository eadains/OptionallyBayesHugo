#!/usr/bin/bash
hugo
git add *
read -p "Commit Message: " m
git commit -m "$m"
git push origin master
git subtree push --prefix public origin gh-pages
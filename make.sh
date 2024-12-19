#!/usr/bin/bash
hugo
git add *
printf "Commit Message:"
read m
git commit -m "$m"
git push origin master
git subtree push --prefix public origin gh-pages

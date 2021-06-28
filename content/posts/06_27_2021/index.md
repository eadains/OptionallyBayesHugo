---
title: A Note on Hugo
date: 2021-06-27
tags: ["hugo", "static-sites"]
slug: hugo_move
summary: My comments on how Hugo compares to Pelican for static site generation
---

I've moved my blog to Hugo for one main reason: I want to start posting work about Julia here, but Pluto.jl only exports in HTML, and Hugo supports that kind of content. In the process of moving, however, I've noticed that Hugo is much nicer to work with than Pelican. Hugo is written in Go, which I have no experience with, but it was very easy to install and use. Pelican, on the other hand, is written in python so I had to create a virtual environment and go through the normal annoyances with dependencies.

Hugo also includes a built in local server so you can work on your site and know what you're getting. It works very smoothly. Writing posts and managing content is also far easier and more organized in Hugo. I can group my posts and their content together all in one folder so everything is nice and neat.

It's much easier also to push my content directly to github pages. I was using a seperate plugin for Pelican, but I've created this shell script now that automatically builds my site, pushes the source to the master branch, and pushes only the static site content to the gh-pages branch.

```bash
#!/usr/bin/bash
hugo
git add *
read -p "Commit Message: " m
git commit -m "$m"
git push origin master
git subtree push --prefix public origin gh-pages
```

Themes are very well supported also, and there are many of them to choose from. Using the PaperMod theme, it was easy to get LaTeX support also, which is important to me.

It's very usable, well documented, and I would recommend it to anyone looking for a static site generator to use for their own personal blog.
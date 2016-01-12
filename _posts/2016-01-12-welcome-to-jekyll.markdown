---
layout: post
title:  "What makes strong Convolutional Go Bot?"
date:   2016-01-12 15:09:18 +0100
excerpt: "ahoj pepo!"
categories: post
---
ahoj pepo jak se mas

{% highlight python %}
import logging
import h5py

class Dataset:
    def __init__(self, filename):
        self.f = h5py.File(filename)
    def iter(self, key='_train', step=50000, head=None):
        num_examples = self.f['x'+key].shape[0]
        if head is not None:
            num_examples = head
# this is a comment

{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/

---
layout: post
title:  "Predicting Go players' strength with Convolutional Neural Nets"
date:   2016-01-15 18:09:18 +0100
excerpt: ""
categories: post
---
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

Ok, as this is my first blog-post here, so I should probably say why did I
start. I want to use this platform to share what I do and maybe get some
feedback for my work as well (I am working on comments, until then, e-mails
are most welcome).

Introduction
----

Now, about the actual subject. As probably everyone around nowadays,
I've been toying with deep learning recently. The domain of my interest
is [computer Go](https://en.wikipedia.org/wiki/Computer_Go),
and since the Go board is essentially a 19x19
bitmap with spatial properties, convolutional neural nets are clear choice.
Most researchers in computer Go are trying to make strong computer programs
(which is a hard problem). Making strong programs means knowing what is the
strongest move in a given position, which makes for simple translation to
the language of convolutional networks. Since we also have a lot of [Go game records](http://gokifu.com)
around, we have some ideas for research papers and fun.

Obviously, this idea has been around for some time now, but recently it
has started to be really hot; see [\[Clark, Storkey 2014\]](clark2014) and
[\[Maddison et al. 2015\]](http://arxiv.org/abs/1412.6564) for starters. Just recently,
a paper from FB research [\[Tian, Zhu 2015\]](http://arxiv.org/abs/1511.06410) has made a great
deal of hype in the news, improving on the previous results. However the FB really
seems to have some people working on this hard, as their bot combining the Monte Carlo
Tree Search (prevalent technology in strong bots nowadays) with the CNN priors
[ended up third](http://www.weddslist.com/kgs/past/119/index.html) on the first
tournament it played. Moreover it lost on time in both games, so lets see
if they can actually win once they improve the time management.

Strength Prediction
----

In the past, I've been working on predicting Go player's strength and playing
style, see [Gostyle Project][gostyle], so I wanted to try how
good are CNN's there. The idea is to give the network a position and teach it
what are the players' strengths instead of predicting the good move.
In Go the strength is measured in 
[kyu/dan ranks](https://en.wikipedia.org/wiki/Go_ranks_and_ratings);
for us, imagine we have a scale of, say, 24 ranks. The ranks have ordering
(rank 1 is stronger player than rank 2), so regression seems like a good choice
to begin with.

### Dataset ###

I used some 71119 games from the [KGS Archives](https://www.gokgs.com/archives.jsp).
Each game has on average cca 190 moves, so in the end, we have almost 14 million
pairs $$(X, y)$$ for training. First we need to make the dataset. The $$y$$'s are clear,
we only need to rescale black and white's strength. For $$X$$'s we need to define
planes, here I used the following 13 planes:

{% highlight python %}
    plane[0] = num_our_liberties == 1
    plane[1] = num_our_liberties == 2
    plane[2] = num_our_liberties == 3
    plane[3] = num_our_liberties >= 4
    plane[4] = num_enemy_liberties == 1
    plane[5] = num_enemy_liberties == 2
    plane[6] = num_enemy_liberties == 3
    plane[7] = num_enemy_liberties >= 4

    plane[8] = empty_points_on_board

    plane[9]  = history_played_before == 1
    plane[10] = history_played_before == 2
    plane[11] = history_played_before == 3
    plane[12] = history_played_before == 4
{% endhighlight %}

This is almost the source code in the tool (below).
Basically the right sides are numpy arrays with some simple domain knowledge (for
instance, planes 9 to 12 show list last 4 moves).
The planes are basically a simple extension of the
[Clark, Storkey 2014][clark2014] paper with the history moves and were proposed by 
[Detlef Schmicker](http://computer-go.org/pipermail/computer-go/2015-December/008324.html).

I used my github project [deep-go-wrap](https://github.com/jmoudrik/deep-go-wrap/) 
which has a tool for making HDF5 datasets from Go game records and all the planes
necessary. Making a dataset is as hard as running:

{% highlight bash %}
cat game_filenames | sort -R | ./make_dataset.py -l ranks -p detlef  out.hdf
{% endhighlight %}

### Network ###

Now, the network. Since this is more like a proof-of-concept experiment, the network
used was fairly simple to have fast training, so I went with a few convolutions with
a small dropout dense layer atop and 2 output neurons for strengths of the players.

 1. convolutional layer 128 times 5x5 filters, ReLU activation
 1. convolutional layer  64 times 3x3 filters, ReLU activation
 1. convolutional layer  32 times 3x3 filters, ReLU activation
 1. convolutional layer   8 times 3x3 filters, ReLU activation
 1. dense layer with 64 neurons, ReLU activation, 0.5 dropout
 1. 2 output neurons, linear activation

For implementation, I used the [keras.io](http://keras.io/) library.
The model [can be found here](/static/20160114/keras_model.py).
The network was trained for just 2 epochs with [Adam](http://arxiv.org/abs/1412.6980v8)
optimizer, because is converged pretty quickly. I have a bigger network
in training (using RMSProp), but it will take some time, so let's have
a look on results in the meantime ;-)

### Results ###

So basically, we now have a network which predicts strength of both players
from a single position (plus history of 4 last moves). This would be really
cool if it worked, as we've previously recommended at least a sample of 10 games
in our [GoStyle webapp][webapp] to predict strength somewhat reliably.

So, how good this really simple first-shot network &mdash; trained for quite a short time,
without any fine-tuning &mdash; performs? And what would a good result be?
Lets have a look on dependency of error (difference between wanted and predicted
value) on move number.

<img src="/static/20160114/err_by_move.png" />

We can clearly see that the error is the highest at the beginning and end.
Because games of beginners usually look the same to games of strong players at the 
beginning (first few moves are usually very similar) the first half is not surprising.
The fact that the error grows so steeply in the end is probably caused by the fact, that
there are only few very long games (usually game takes about 250-300 moves).
Before discussing whether are these numbers any good, let's have a look on another graph,
error by rank:

<img src="/static/20160114/err_by_rank.png" />

This graph basically shows, that hardest players to be predicted are both very strong and
very weak players. A graph of a predictor which would predict just the middle class would
look like a letter "V" with the minimum in the middle. This graph has more of a "U" shape,
which is good, because it means, that the network is not only utilising statistical
distribution of target $$y$$'s but also some knowledge. Comparing with the naive V predictor
is also interesting in terms of error. Were the 24 $$y$$ values distributed
[uniformly randomly][uniform]
the standard deviation of the always-the-middle V predictor would be `

$$\sigma^2 = Var(U(0,24)) = \frac{24^2}{12} = 6.93^2$$

On average, the network has $$RMSE$$ (root of mean square error) or 4.66. The $$RMSE$$ has
the nice property that (under certain assumptions), it is estimate of the $$\sigma$$,
so we can say, that the network actually does something useful.


### Comparison With Prior Work ###

In my recent paper [Evaluating Go game records for prediction of player attributes][eval2],
different features were able (with a given good model) to predict the strength with
the following $$RMSE$$:

* 2.788 Pattern feature
* 5.765 Local sequences
* 5.818 Border distance
* 5.904 Captured stones
* 6.792 Win/Loss statistics
* 5.116 Win/Loss points

The results in the paper had a slightly bigger domain of 26 ranks instead of 24, but roughly
it is comparable. So 4.66 of our brave new deep network is better than all but
the dominating feature, and this from just one game position with history of size 4.
Cool indeed!


### What next? ###

 * You have some ideas? I do. Stay tuned!

<!-- ref [Jekyll docs][jekyll-docs] -->

[jekyll]: http://jekyllrb.com/
[clark2014]: http://arxiv.org/abs/1412.3409
[webapp]: http://gostyle.j2m.cz/webapp.html
[gostyle]: http://gostyle.j2m.cz
[eval]: http://dx.doi.org/10.1109/CIG.2015.7317909
[eval2]: http://arxiv.org/abs/1512.08969 
[uniform]: https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)


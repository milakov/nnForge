Galaxy Zoo
==========

[Galaxy Zoo](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) was a data analysis challenge at [Kaggle](https://www.kaggle.com). Contestants were asked to predict how humans answered the series of classification questions given the image of the galaxy. This solution took the 2nd place in the challenge.

nnForge
-------

The code uses nnForge library. Please read [corresponding Readme](../../README.md) first.

Input data
----------

Download *images_training_rev1.zip*, *images_test_rev1.zip*, and *training_solutions_rev1.zip* from [Data page](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data).

Extract input files from archives and store them in the galaxy_zoo input directory in the following way:

	galaxy_zoo/
		images_training_rev1/
			100008.jpg
			...
			999967.jpg
		images_test_rev1/
			100018.jpg
			...
			999996.jpg
		training_solutions_rev1.csv

Setup
-----

	./galaxy_zoo prepare_training_data
	./galaxy_zoo generate_output_normalizer
	./galaxy_zoo randomize_data
	./galaxy_zoo prepare_testing_data
	./galaxy_zoo create

Train
-----

	./galaxy_zoo train -N 15

This training will take ~3 days on GeForce GTX Titan (and months on CPU). You might want to try training a single network first.
Instead of training the network you might download [the archive](https://drive.google.com/uc?id=0B2hfQbOo3RqBVE91S2ZmMFpKNkU&export=download) with the networks I trained.
Regardless of the way you get trained networks you should have them in the batch sub-directory of working galaxy_zoo directory:

	galaxy_zoo/
		batch/
			ann_trained_000.data
			...
			ann_trained_014.data

Test
----

	./galaxy_zoo test

This will take ~10 hours on GeForce GTX Titan (and weeks on CPU) and will generate output.csv with probabilities predicted for each testing image.

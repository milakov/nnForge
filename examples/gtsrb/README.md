GTSRB
=====

GTSRB stands for German Traffic Sign Recognition Benchmark. All the info is available at [INI website](http://benchmark.ini.rub.de/?section=gtsrb).

J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In _Proceedings of the IEEE International Joint Conference on Neural Networks_, pages 1453–1460. 2011.

Input data
----------

Download _Images and annotations_ both for _official GTSRB training set_ and _official GTSRB test set_, and also _Extended annotations including class ids_ for the latter from [Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads).

Extract input files from archives and store them in the gtsrb input directory in the following way:

	gtsrb/
		Final_Training/
			Images/
				00000/
					00000_00000.ppm
					...
					00006_00029.ppm
					GT-00000.csv
				...
				00042/
					00000_00000.ppm
					...
					00007_00029.ppm
					GT-00042.csv
		Final_Test/
			Images/
				00000.ppm
				...
				12629.ppm
				GT-final_test.csv

GT-final_test.csv in Final_Test should be the one with class IDs, from _Extended annotations including class ids_ archive. 

Train
-----

	./gtsrb prepare_training_data
	./gtsrb randomize_data
	./gtsrb create
	./gtsrb train_batch -N 10
	
Validate
--------

	./gtsrb validate_batch

I got about 0.3% error rate on validation data. You will likely get similar results.

ImageNet
========

Input data
----------

Input data should be organized in the following folder structure:

	input_data_imagenet/
		ILSVRC2012_img_train/
			n01440764/
				n01440764_10026.JPEG
				n01440764_10027.JPEG
				...
				n01440764_9981.JPEG
			n01443537/
				n01443537_10007.JPEG
				n01443537_10014.JPEG
				...
				n01443537_9977.JPEG
		ILSVRC2012_img_val/
			ILSVRC2012_val_00000001.JPEG
			ILSVRC2012_val_00000002.JPEG
			...
			ILSVRC2012_val_00050000.JPEG

Working folder should contain cls_class_info.txt file. See misc_files/generate_cls_class_info.m, it will generate the file for you.

Train
-----

	./imagenet prepare_training_data
	./imagenet generate_input_normalizer
	./imagenet train
	
Training will take a couple of weeks on modern GPU and will give you about 14% Top-5 error on single image inference.

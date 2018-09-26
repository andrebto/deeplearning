from keras.preprocessing import image
from keras.applications import densenet, inception_resnet_v2, inception_v3, resnet50, vgg16, vgg19, xception
from keras.models import Model, load_model
import numpy as np

#Get pretrained network
def get_network(network):

	sizes = {'inceptionv3':inception_v3.InceptionV3, 'densenet':densenet.DenseNet121,
			'xception':xception.Xception,'resnet50':resnet50.ResNet50,
			'inception_resnet_v2':inception_resnet_v2.InceptionResNetV2,
			'vgg16':vgg16.VGG16, 'vgg19':vgg19.VGG19}

	return sizes[network]

#Get preprocess method according to network used
def get_preprocess_input(network):

	sizes = {'inceptionv3':inception_v3.preprocess_input, 'densenet':densenet.preprocess_input,
				'xception':xception.preprocess_input,'resnet50':resnet50.preprocess_input,
				'inceptionresnetv2':inception_resnet_v2.preprocess_input,
				'vgg16':vgg16.preprocess_input, 'vgg19':vgg19.preprocess_input}

	return sizes[network]


class Extractor():

    def __init__(self, network = 'resnet50', layer_name = 'avg_pool', weights = None):

        self.weights = weights
        self.network = network

        if(weights is None):
            #Get network pretrained model
            base_model = get_network(network)(weights='imagenet', include_top=True)
        else:
            base_model = load_model(weights)

        #Find a layer in model with the same name informed.
        output = None
        for layer in base_model.layers:
            if layer.name == layer_name:
                output = layer.output

        assert output is not None, "Layer " + layer_name + " not found in model"

        # We'll extract features at the informed layer.
        self.model = Model(inputs=base_model.input, outputs=output)


    def extract(self, image_path, image_size, flat = False):

        #Load and preprocess image
        img = image.load_img(image_path, target_size=(image_size, image_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        #Get network preprocess method
        x = get_preprocess_input(self.network)(x)

        # Get the prediction.
        features = self.model.predict(x)

        #Remove single-dimensional entries from the shape of an array
        features = np.squeeze(features)

        #Flatten features
        if((features.ndim > 1) and (flat)):
            features = features.flatten()

        return features

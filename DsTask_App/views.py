from django.shortcuts import render
from joblib import load
from simpletransformers.language_representation import RepresentationModel
from simpletransformers.config.model_args import ModelArgs


def first_page(request):
	return render(request, './index.html')


def predict(request):
	comment = request.GET['Comment']
	model = load('C:\\Users\\HONOR\\Desktop\\Дима\\Программирвоание\\Работа\\Тестовое задание\\aclImdb\\django_service\\DsTask\\DsTask_App\\LogRegression.joblib')

	embeding_model = RepresentationModel(
		model_type="bert",
		model_name="bert-base-uncased",
		use_cuda=False
	)

	sentence_vectors = embeding_model.encode_sentences([comment], combine_strategy='mean')
	result = model.predict(sentence_vectors)[0]
	

	return render(request, './predict.html', {'result' : result})
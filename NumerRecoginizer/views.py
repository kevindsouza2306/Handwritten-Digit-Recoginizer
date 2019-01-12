from django.shortcuts import render
import re
import base64
from predict_number import PredictNumber
from keras import backend as K

# Create your views here.
def simple_upload(request):
    if request.method == 'POST' and request.POST.get('myfile'):
        dataUrlPattern = re.compile('data:image/(png|jpeg);base64,(.*)$')
        image_data = request.POST.get('myfile')
        image_data = dataUrlPattern.match(image_data).group(2)
        image_data = image_data.encode()
        image_data = base64.b64decode(image_data)

        with open('media/image.jpg', 'wb') as f:
            f.write(image_data)
        predicted_number = PredictNumber()
        K.clear_session()

        return render(request, 'NumerRecoginizer/homePage.html', {
                'uploaded_file_url': predicted_number
            })
    return render(request, 'NumerRecoginizer/homePage.html')

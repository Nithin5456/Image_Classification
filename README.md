

 Image Classification Project  

 Project Overview  
This project is a machine learning-based web application that classifies images into one of five categories: Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli. The model uses SVM (Support Vector Machine) and processes images through cropping and wavelet transformation to improve accuracy.  

 Features  
- Upload an image for classification.  
- Displays the predicted category of the uploaded image.  
- Simple and user-friendly web interface built with Django.  

 Technologies Used  
- Backend: Django (Python)  
- Frontend: HTML, CSS  
- Machine Learning Model: Support Vector Machine (SVM)  
- Image Preprocessing: OpenCV, NumPy  
 

 Setup Instructions  

1. Install dependencies:  
   bash  
   pip install -r requirements.txt  
     

2. Place the trained model and class dictionary:  
   - Save the model file (model.pkl) and class dictionary (class_dict.json) in the model folder.  

3. Run database migrations (if any):  
   bash  
   python manage.py migrate  
     

4. Start the Django development server:  
   bash  
   python manage.py runserver  
     

5. Open the application in your browser at http://127.0.0.1:8000.  

 Usage  
1. Navigate to the homepage.  
2. Upload an image file.  
3. Click the "Classify" button.  
4. View the predicted category.  
  

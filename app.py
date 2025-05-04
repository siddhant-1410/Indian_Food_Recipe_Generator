from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, jsonify
import os
from werkzeug.utils import secure_filename
from modules.dish_classifier import predict_image
from modules.recipe_generator import generate_recipe, translate_text
from modules.recomendation_inference import load_model_and_data, get_dish_recommendations
from dotenv import load_dotenv
import os

#API Key loading -
load_dotenv()
api_key = os.getenv("API_KEY")

# Flask Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-replace-this'
app.config['DEBUG'] = True

# File Upload Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PDF_FOLDER'] = 'pdfs'
app.config['MODELS_FOLDER'] = 'models'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# API Configuration
GEMINI_API_KEY = api_key  # Replace with your actual API key

# Load recommendation model and data
model, model_data, cleaned_data = load_model_and_data()

# Ensure required directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['PDF_FOLDER'], app.config['MODELS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

def allowed_file(filename):
    """Check if the file extension is allowed"""
    print("[allowed_file] Checking file extension for:", filename)
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    print("[index] Rendering index.html")
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_dish():
    """Handle image upload and classification"""
    print("[classify_dish] Received request to classify image")
    
    if 'image' not in request.files:
        flash('No image part')
        print("[classify_dish] No image part in request")
        return redirect(url_for('index'))
    
    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        print("[classify_dish] No selected file")
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"[classify_dish] Saved uploaded file to {filepath}")
        
        try:
            dish_name, confidence = predict_image(filepath)
            print(f"[classify_dish] Prediction: {dish_name}, Confidence: {confidence}")
            return render_template(
                'classification.html', 
                dish_name=dish_name, 
                confidence=round(confidence * 100, 2),
                image_filename=filename
            )
        except Exception as e:
            flash(f'Error classifying image: {str(e)}')
            print(f"[classify_dish] Error: {str(e)}")
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload JPG, JPEG or PNG.')
        print("[classify_dish] Invalid file type")
        return redirect(url_for('index'))

@app.route('/generate_recipe', methods=['POST'])
def get_recipe():
    """Generate recipe based on dish name"""
    print("[get_recipe] Received request to generate recipe")
    
    dish_name = request.form.get('dish_name')
    if not dish_name:
        flash('No dish name provided')
        print("[get_recipe] No dish name provided")
        return redirect(url_for('index'))
    
    try:
        recipe = generate_recipe(
            dish_name=dish_name,
            pdf_folder=app.config['PDF_FOLDER'],
            api_key=GEMINI_API_KEY
        )
        print(f"[get_recipe] Recipe generated for: {dish_name}")
        
        return render_template(
            'recipe.html', 
            dish_name=dish_name, 
            recipe=recipe
        )
    except Exception as e:
        flash(f'Error generating recipe: {str(e)}')
        print(f"[get_recipe] Error: {str(e)}")
        return redirect(url_for('index'))

@app.route('/recommend_dishes', methods=['POST'])
def recommend_dishes():
    """Get dish recommendations based on a dish name"""
    print("[recommend_dishes] Received request for dish recommendations")
    
    dish_name = request.form.get('dish_name')
    if not dish_name:
        flash('No dish name provided')
        print("[recommend_dishes] No dish name provided")
        return redirect(url_for('index'))
    
    try:
        # Default to 5 recommendations if not specified
        n_recommendations = int(request.form.get('n_recommendations', 5))
        
        # Get recommendations using the loaded model and data
        recommendations = get_dish_recommendations(
            dish_name=dish_name,
            model=model,
            model_data=model_data,
            cleaned_data=cleaned_data,
            n_recommendations=n_recommendations
        )
        
        if not recommendations:
            flash(f'No recommendations found for {dish_name}')
            print(f"[recommend_dishes] No recommendations found for: {dish_name}")
            return redirect(url_for('index'))
        
        print(f"[recommend_dishes] Found {len(recommendations)} recommendations for: {dish_name}")
        
        # Get image filename if it was uploaded
        image_filename = request.form.get('image_filename', None)
        
        return render_template(
            'recommendations.html',
            dish_name=dish_name,
            recommendations=recommendations,
            image_filename=image_filename
        )
    except Exception as e:
        flash(f'Error getting recommendations: {str(e)}')
        print(f"[recommend_dishes] Error: {str(e)}")
        return redirect(url_for('index'))

@app.route('/translate_recipe', methods=['POST'])
def translate_recipe():
    """Translate recipe to Hindi"""
    print("[translate_recipe] Received request to translate recipe")
    
    recipe_text = request.form.get('recipe')
    if not recipe_text:
        return jsonify({'error': 'No recipe text provided'}), 400
    
    try:
        translated_recipe = translate_text(
            recipe_text,
            model_path='best_model'
        )
        print("[translate_recipe] Recipe translated successfully")
        
        return jsonify({
            'translated_recipe': translated_recipe
        })
    except Exception as e:
        print(f"[translate_recipe] Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    print(f"[uploaded_file] Sending file: {filename}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    print("[too_large] Uploaded file too large")
    flash('File is too large. Maximum size is 16MB.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("[main] Starting Flask app")
    app.run(debug=True)
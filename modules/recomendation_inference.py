import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

def load_model_and_data():
    """
    Load the trained model and processed data
    """
    try:
        # Load the trained model
        with open(r"C:\Users\karti\OneDrive\Desktop\presentable_recipe_app_cpy_ori_dlds - Copy\models\model.pickle", "rb") as f:
            model = pickle.load(f)
        
        # Load the preprocessed data that was used for modeling
        model_data = pd.read_csv(r"C:\Users\karti\OneDrive\Desktop\presentable_recipe_app_cpy_ori_dlds - Copy\models\model_data.csv", index_col=0)
        
        # Load the cleaned data for displaying full information about dishes
        cleaned_data = pd.read_csv(r"C:\Users\karti\OneDrive\Desktop\presentable_recipe_app_cpy_ori_dlds - Copy\models\cleaned_data.csv")
        
        print("Model and data loaded successfully!")
        return model, model_data, cleaned_data
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the model.pickle, model_data.csv, and cleaned_data.csv files are in the current directory.")
        return None, None, None

def get_dish_recommendations(dish_name, model, model_data, cleaned_data, n_recommendations=5):
    """
    Get dish recommendations based on a dish name
    
    Parameters:
    - dish_name: Name of the dish to get recommendations for
    - model: Trained NearestNeighbors model
    - model_data: Processed data used for the model
    - cleaned_data: Original cleaned data with all features
    - n_recommendations: Number of recommendations to return
    
    Returns:
    - List of recommended dishes with details
    """
    # Convert dish_name to lowercase to match the index in model_data
    dish_name = dish_name.lower().strip()
    
    # Check if the dish exists in our dataset
    if dish_name not in model_data.index:
        print(f"Dish '{dish_name}' not found in the dataset.")
        print("Available dishes include:", ", ".join(model_data.index[:10]), "...")
        return []
    
    # Get n+1 neighbors (including the input dish itself)
    distances, indices = model.kneighbors(
        model_data.loc[[dish_name]], 
        n_neighbors=n_recommendations+1
    )
    
    # Flatten the arrays
    distances = distances.flatten()
    indices = indices.flatten()
    
    # Create a list to store recommended dishes
    recommendations = []
    
    # Skip the first result if it's the input dish itself
    for i in range(len(indices)):
        recommended_dish_name = model_data.index[indices[i]]
        
        # Skip if it's the same as the input dish
        if recommended_dish_name == dish_name:
            continue
            
        # Get full details from cleaned_data
        dish_details = cleaned_data[cleaned_data['name'] == recommended_dish_name].iloc[0].to_dict()
        
        # Add similarity score (convert distance to similarity)
        dish_details['similarity_score'] = 1 - distances[i]
        
        recommendations.append(dish_details)
        
        # Stop once we have n recommendations
        if len(recommendations) >= n_recommendations:
            break
    
    return recommendations

def display_dish_details(dish_name, cleaned_data):
    """
    Display detailed information about a specific dish
    """
    dish_name = dish_name.lower().strip()
    dish_info = cleaned_data[cleaned_data['name'] == dish_name]
    
    if len(dish_info) == 0:
        print(f"Dish '{dish_name}' not found in the dataset.")
        return
    
    dish_info = dish_info.iloc[0]
    
    print(f"\n{'='*50}")
    print(f"DISH: {dish_info['name'].upper()}")
    print(f"{'='*50}")
    print(f"Diet: {dish_info['diet']}")
    print(f"Flavor Profile: {dish_info['flavor_profile']}")
    print(f"Course: {dish_info['course']}")
    print(f"State: {dish_info['state']}")
    print(f"Region: {dish_info['region']}")
    
    if dish_info['prep_time'] != -1:
        print(f"Prep Time: {dish_info['prep_time']} minutes")
    if dish_info['cook_time'] != -1:
        print(f"Cook Time: {dish_info['cook_time']} minutes")
    
    print(f"\nIngredients: {dish_info['ingredients']}")
    print(f"{'='*50}")

def display_recommendations(recommendations):
    """
    Display the recommended dishes in a formatted way
    """
    print("\nRECOMMENDED DISHES:")
    print("-" * 50)
    
    for i, dish in enumerate(recommendations, 1):
        print(f"{i}. {dish['name'].upper()} (Similarity: {dish['similarity_score']:.2f})")
        print(f"   Diet: {dish['diet']} | Flavor: {dish['flavor_profile']} | Course: {dish['course']}")
        print(f"   State: {dish['state']} | Region: {dish['region']}")
        print("-" * 50)

def interactive_mode():
    """
    Run the recommendation system in interactive mode
    """
    # Load model and data
    model, model_data, cleaned_data = load_model_and_data()
    if model is None or model_data is None or cleaned_data is None:
        return
    
    print("\nWelcome to the Indian Food Recommendation System!")
    print("=" * 60)
    
    while True:
        print("\nChoose an option:")
        print("1. Get recommendations for a dish")
        print("2. Display details for a specific dish")
        print("3. Show random dish")
        print("4. Quit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            dish_name = input("\nEnter the name of a dish: ")
            n_recommendations = input("How many recommendations would you like? (default: 5): ")
            
            try:
                n_recommendations = int(n_recommendations)
            except:
                n_recommendations = 5
                
            recommendations = get_dish_recommendations(
                dish_name, 
                model, 
                model_data, 
                cleaned_data, 
                n_recommendations
            )
            
            if recommendations:
                display_recommendations(recommendations)
                
        elif choice == "2":
            dish_name = input("\nEnter the name of a dish: ")
            display_dish_details(dish_name, cleaned_data)
            
        elif choice == "3":
            # Get a random dish
            random_dish = cleaned_data['name'].sample().iloc[0]
            print(f"\nRandom dish selected: {random_dish}")
            display_dish_details(random_dish, cleaned_data)
            
        elif choice == "4":
            print("\nThank you for using the Indian Food Recommendation System!")
            break
            
        else:
            print("\nInvalid choice! Please try again.")

def main():
    """
    Main function to run the recommendation system
    """
    interactive_mode()

if __name__ == "__main__":
    main()
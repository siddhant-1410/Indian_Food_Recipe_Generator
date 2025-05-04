document.addEventListener('DOMContentLoaded', function() {
    // Group all recipe metadata at the top into a nice box
    const recipeContent = document.querySelector('.recipe-content');
    
    if (recipeContent) {
        const metaItems = recipeContent.querySelectorAll('.recipe-meta-item');
        
        if (metaItems.length > 0) {
            // Create a container for all metadata
            const infoBox = document.createElement('div');
            infoBox.className = 'recipe-info-box';
            
            // Move all meta items into the info box with better styling
            metaItems.forEach(item => {
                const label = item.querySelector('.recipe-meta-label').textContent.replace(':', '');
                const value = item.querySelector('.recipe-meta-value').textContent;
                
                const infoItem = document.createElement('div');
                infoItem.className = 'recipe-info-item';
                infoItem.innerHTML = `
                    <span class="recipe-info-label">${label}</span>
                    <span class="recipe-info-value">${value}</span>
                `;
                
                infoBox.appendChild(infoItem);
                item.remove();
            });
            
            // Insert the info box at the beginning of the recipe content
            recipeContent.insertBefore(infoBox, recipeContent.firstChild);
        }
        
        // Add a separator between ingredients and instructions if they exist
        const ingredientList = recipeContent.querySelector('.ingredient-list');
        const instructionList = recipeContent.querySelector('.instruction-list');
        
        if (ingredientList && instructionList) {
            const separator = document.createElement('hr');
            separator.className = 'my-4';
            
            // Insert the separator between the ingredients and instructions
            if (ingredientList.nextElementSibling !== instructionList) {
                recipeContent.insertBefore(separator, instructionList.previousElementSibling);
            }
        }
    }
});
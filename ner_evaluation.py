# simple_evaluation.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from filipino_food_config import FilipinoFoodNER, FILIPINO_FOODS

def nerEvaluator():
    """Simple NER evaluation with 117 test samples (67 Filipino foods + 50 non-foods)."""
    
    # Load model
    print("Loading NER model...")
    ner_model = FilipinoFoodNER()
    nlp = ner_model.load_model_with_ruler()
    
    # Test data: 67 Filipino foods + 50 non-food sentences
    filipino_food_sentences = [f"I love eating {food}." for food in FILIPINO_FOODS]
    
    non_food_sentences = [
        "I went to the store today.", "John works at Microsoft.", "The meeting is in Manila.",
        "She bought a new car.", "The weather is nice.", "Pizza and pasta for dinner.",
        "I love sushi and ramen.", "Coffee and donuts this morning.", "Basketball game tonight.",
        "Reading a good book.", "Maria lives in Cebu.", "The conference starts at 9 AM.",
        "Google launched a new product.", "I visited New York last year.", "The movie was great.",
        "He graduated from UP.", "Apple released the iPhone.", "Traffic is heavy today.",
        "I need to buy groceries.", "The flight to Japan is delayed.", "She works as a doctor.",
        "The restaurant is expensive.", "I downloaded a new app.", "Christmas is next month.",
        "The exam is on Friday.", "Samsung makes good phones.", "I called my mother yesterday.",
        "The concert was amazing.", "Tesla stock went up.", "I'm reading Harry Potter.",
        "The gym is closed today.", "Facebook changed its name.", "I bought new shoes.",
        "The wedding is in December.", "Netflix has good shows.", "I learned to cook pasta.",
        "The beach was crowded.", "Amazon delivered my package.", "I took a taxi home.",
        "The hotel room was clean.", "YouTube has funny videos.", "I planted some flowers.",
        "The airplane was delayed.", "Instagram updated its features.", "I visited my grandmother.",
        "The library is quiet.", "Microsoft Office is useful.", "I watched a documentary.",
        "The park is beautiful.", "Twitter has breaking news.", "I bought concert tickets.",
        "The school is nearby.", "LinkedIn job posting.", "I studied mathematics."
    ]
    
    # Combine test data (67 Filipino foods + 50 completely non-food sentences)
    all_sentences = filipino_food_sentences + non_food_sentences
    expected_labels = ["FILIPINO_FOOD"] * len(filipino_food_sentences) + ["OTHER"] * len(non_food_sentences)
    
    print(f"Testing {len(all_sentences)} sentences...")
    print(f"- Filipino food sentences: {len(filipino_food_sentences)} (all items from FILIPINO_FOODS)")
    print(f"- Non-food sentences: {len(non_food_sentences)} (no Filipino foods)")
    print(f"Total test samples: {len(all_sentences)}")
    
    # Evaluate model
    predicted_labels = []
    detailed_results = []
    
    for i, (sentence, expected) in enumerate(zip(all_sentences, expected_labels)):
        doc = nlp(sentence)
        filipino_foods_found = [ent.text for ent in doc.ents if ent.label_ == "FILIPINO_FOOD"]
        
        # Predict label
        if filipino_foods_found:
            predicted = "FILIPINO_FOOD"
        else:
            predicted = "OTHER"
        
        predicted_labels.append(predicted)
        
        # Store detailed results
        detailed_results.append({
            'sentence': sentence,
            'expected': expected,
            'predicted': predicted,
            'correct': expected == predicted,
            'found_foods': filipino_foods_found
        })
        
        # Show progress every 20 sentences
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(all_sentences)} sentences...")
    
    # Calculate metrics
    cm = confusion_matrix(expected_labels, predicted_labels, labels=["FILIPINO_FOOD", "OTHER"])
    precision, recall, f1, support = precision_recall_fscore_support(
        expected_labels, predicted_labels, labels=["FILIPINO_FOOD", "OTHER"]
    )
    accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
    
    # Print results
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total test samples: {len(all_sentences)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (Filipino food detected): {cm[0][0]}")
    print(f"  True Negatives (non-food correctly identified): {cm[1][1]}")
    print(f"\nPer-class metrics:")
    print(f"  FILIPINO_FOOD - Precision: {precision[0]:.3f}, Recall: {recall[0]:.3f}, F1: {f1[0]:.3f}")
    print(f"  OTHER - Precision: {precision[1]:.3f}, Recall: {recall[1]:.3f}, F1: {f1[1]:.3f}")
    
    # Show any errors
    errors = [r for r in detailed_results if not r['correct']]
    if errors:
        print(f"\nErrors found: {len(errors)}")
        for error in errors:
            print(f"  '{error['sentence']}'")
            print(f"    Expected: {error['expected']}, Got: {error['predicted']}")
    else:
        print(f"\nNo errors found! Perfect classification.")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Confusion Matrix
    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=["FILIPINO_FOOD", "OTHER"],
               yticklabels=["FILIPINO_FOOD", "OTHER"])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Metrics Bar Chart
    plt.subplot(2, 2, 2)
    x = np.arange(3)
    width = 0.35
    
    plt.bar(x - width/2, [precision[0], recall[0], f1[0]], width, 
           label='FILIPINO_FOOD', alpha=0.8, color='orange')
    plt.bar(x + width/2, [precision[1], recall[1], f1[1]], width, 
           label='OTHER', alpha=0.8, color='blue')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    plt.xticks(x, ['Precision', 'Recall', 'F1-Score'])
    plt.legend()
    plt.ylim(0, 1.1)
    
    # Add values on bars
    for i in range(3):
        plt.text(i - width/2, [precision[0], recall[0], f1[0]][i] + 0.02, 
                f'{[precision[0], recall[0], f1[0]][i]:.3f}', ha='center', fontsize=9)
        plt.text(i + width/2, [precision[1], recall[1], f1[1]][i] + 0.02, 
                f'{[precision[1], recall[1], f1[1]][i]:.3f}', ha='center', fontsize=9)
    
    # Accuracy visualization
    plt.subplot(2, 2, 3)
    correct_count = sum(1 for r in detailed_results if r['correct'])
    incorrect_count = len(detailed_results) - correct_count
    
    plt.pie([correct_count, incorrect_count], 
           labels=[f'Correct\n({correct_count})', f'Incorrect\n({incorrect_count})'],
           colors=['lightgreen', 'lightcoral'], 
           autopct='%1.1f%%',
           startangle=90)
    plt.title(f'Overall Accuracy: {accuracy:.1%}')
    
    # Classification report
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    report_text = f"""
Classification Report:

FILIPINO_FOOD:
  Precision: {precision[0]:.3f}
  Recall:    {recall[0]:.3f} 
  F1-Score:  {f1[0]:.3f}
  Support:   {support[0]}

OTHER:
  Precision: {precision[1]:.3f}
  Recall:    {recall[1]:.3f}
  F1-Score:  {f1[1]:.3f}
  Support:   {support[1]}

Overall:
  Accuracy:  {accuracy:.3f}
  Total:     {len(all_sentences)} samples
    """
    
    plt.text(0.1, 0.5, report_text, fontsize=10, fontfamily='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('simple_filipino_food_ner_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as: simple_filipino_food_ner_evaluation.png")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'errors': errors,
        'total_samples': len(all_sentences)
    }

if __name__ == "__main__":
    results = nerEvaluator()
    print("\nEvaluation complete!")
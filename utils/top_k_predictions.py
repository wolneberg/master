def calculate_accuracy(predictions, k=1):
    correct_predictions = 0
    top_k_correct_predictions = 0
    total_predictions = len(predictions)
    for actual, pred in predictions.items():
        x = actual.deref()
        if x.numpy()[0] == pred.numpy()[0][0]:
            correct_predictions += 1
        
        # Check if the actual label is in the top k predictions
        if x[0] in pred.numpy()[0][:k]:
            top_k_correct_predictions += 1
    
    accuracy = (correct_predictions / total_predictions) * 100
    top_k_accuracy = (top_k_correct_predictions / total_predictions) * 100
    return accuracy, top_k_accuracy

def calculate_accuracy_onnx(predictions, k=1):
    correct_predictions = 0
    top_k_correct_predictions = 0
    total_predictions = len(predictions)
    for actual, pred in predictions.items():
        x = actual.deref()
        # # print(x[0])
        # print(pred.numpy())
        # print(x.numpy())
        # print(pred[0])
        # print(pred[0][0])
        if x.numpy()[0] == pred.numpy()[0]:
            correct_predictions += 1
        
        # Check if the actual label is in the top k predictions
        if x in pred.numpy()[:k]:
            top_k_correct_predictions += 1
    
    accuracy = (correct_predictions / total_predictions) * 100
    top_k_accuracy = (top_k_correct_predictions / total_predictions) * 100
    return accuracy, top_k_accuracy
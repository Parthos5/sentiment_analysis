function predictSentiment() {
    var inputText = document.getElementById("inputText").value;

    // Send input text to the API endpoint
    fetch('http://127.0.0.1:5000/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: inputText })
    })
    .then(res => res.json())
    .then(data => {
        // Display the predicted sentiment
        document.getElementById("predictionResult").innerText = `Predicted sentiment: ${data.sentiment}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    // Grab the user inputs
    const payload = {
        precipitation: parseFloat(document.getElementById('precipitation').value),
        tmean: parseFloat(document.getElementById('tmean').value),
        pet: parseFloat(document.getElementById('pet').value),
        wb: parseFloat(document.getElementById('wb').value)
    };

    // Send POST request to the API Handshake
    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        // Display Predictions in the Results section
        if (data.status === 'success') {
            document.getElementById('results').style.display = 'block';
            document.getElementById('targetName').innerText = data.metadata.target;
            document.getElementById('predTransformer').innerText = data.predictions.transformer.toFixed(2);
            document.getElementById('predLSTM').innerText = data.predictions.lstm.toFixed(2);
            document.getElementById('predRF').innerText = data.predictions.random_forest.toFixed(2);
        } else {
            alert('Error: ' + (data.error || 'Unknown error occurred on the backend'));
        }
    } catch (error) {
        console.error('Error fetching prediction:', error);
        alert('Failed to connect to the prediction API at http://127.0.0.1:5000/predict.');
    }
});

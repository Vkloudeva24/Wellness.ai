document.getElementById("healthForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData);

    try {
        const response = await fetch("http://127.0.0.1:8501/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });

        if (response.ok) {
            const result = await response.json();
            document.getElementById("healthStatus").textContent = result.health_status;
            document.getElementById("recommendation").textContent = result.recommendation;
            document.getElementById("output").style.display = "block";
        } else {
            alert("Error fetching prediction. Please try again.");
        }
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred. Please check the console for details.");
    }
});

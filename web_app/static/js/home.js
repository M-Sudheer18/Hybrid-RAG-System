document.addEventListener("DOMContentLoaded", async () => {
    const countElement = document.getElementById("docCount");

    if (!countElement) {
        return;
    }

    try {
        const response = await fetch("/api/documents/");

        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }

        const data = await response.json();

        countElement.textContent = data.total_count;

    } catch (error) {
        console.error("Failed to fetch document count:", error);
        countElement.textContent = "Unavailable";
    }
});
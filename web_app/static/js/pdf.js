document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const elements = {
        docCount: document.getElementById('documentCount'),
        docsLoading: document.getElementById('documentsLoading'),
        docsEmpty: document.getElementById('documentsEmpty'),
        docsList: document.getElementById('documentsList'),
        
        form: document.getElementById('uploadForm'),
        dropZone: document.getElementById('dropZone'),
        fileInput: document.getElementById('pdfFile'),
        selectedFile: document.getElementById('selectedFile'),
        fileName: document.getElementById('selectedFileName'),
        fileSize: document.getElementById('selectedFileSize'),
        removeBtn: document.getElementById('removeFile'),
        uploadBtn: document.getElementById('uploadButton'),
        uploadStatus: document.getElementById('uploadStatus')
    };

    let currentFile = null;

    // --- Document Listing ---
    async function fetchDocuments() {
        try {
            const response = await fetch('/api/documents/');
            const data = await response.json();
            
            elements.docsLoading.hidden = true;
            
            // Assuming your API returns a list or an object containing the list
            const docs = Array.isArray(data) ? data : (data.documents || []);
            
            elements.docCount.textContent = docs.length;

            if (docs.length === 0) {
                elements.docsEmpty.hidden = false;
                elements.docsList.hidden = true;
            } else {
                elements.docsEmpty.hidden = true;
                elements.docsList.hidden = false;
                renderDocuments(docs);
            }
        } catch (error) {
            console.error('Error fetching documents:', error);
            elements.docsLoading.textContent = 'Error loading documents. Please try refreshing.';
        }
    }

    function renderDocuments(docs) {
        elements.docsList.innerHTML = docs.map(doc => {
            const statusClass = doc.status === 'completed' ? 'status-completed' : 
                              doc.status === 'failed' ? 'status-failed' : 'status-processing';
            
            const chunkCount = doc.chunk_count !== undefined ? `${doc.chunk_count} chunks` : '';
            
            return `
                <div class="document-card">
                    <div class="doc-name">${doc.file_name}</div>
                    <div class="doc-meta">
                        <span>${new Date(doc.ingestion_time).toLocaleDateString()}</span>
                        <span>${chunkCount}</span>
                    </div>
                    <div class="doc-status ${statusClass}">
                        ${doc.status ? doc.status.charAt(0).toUpperCase() + doc.status.slice(1) : 'Indexed'}
                    </div>
                </div>
            `;
        }).join('');
    }

    // --- File Upload Handling ---
    function handleFileSelection(file) {
        if (!file || file.type !== 'application/pdf') {
            showStatus('Please select a valid PDF file.', 'error');
            return;
        }

        currentFile = file;
        elements.dropZone.hidden = true;
        elements.selectedFile.hidden = false;
        elements.fileName.textContent = file.name;
        elements.fileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
        elements.uploadBtn.disabled = false;
        elements.uploadStatus.hidden = true;
    }

    function resetUpload() {
        currentFile = null;
        elements.fileInput.value = '';
        elements.dropZone.hidden = false;
        elements.selectedFile.hidden = true;
        elements.fileName.textContent = '';
        elements.fileSize.textContent = '';
        elements.uploadBtn.disabled = true;
        elements.uploadStatus.hidden = true;
    }

    function showStatus(message, type) {
        elements.uploadStatus.textContent = message;
        elements.uploadStatus.className = `upload-status status-${type}`;
        elements.uploadStatus.hidden = false;
    }

    // Event Listeners for Drag and Drop
    elements.dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.dropZone.classList.add('drag-active');
    });

    elements.dropZone.addEventListener('dragleave', () => {
        elements.dropZone.classList.remove('drag-active');
    });

    elements.dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.dropZone.classList.remove('drag-active');
        if (e.dataTransfer.files.length) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    // Event Listeners for Click Upload
    elements.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileSelection(e.target.files[0]);
        }
    });

    elements.removeBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        resetUpload();
    });

    // Form Submission
    elements.form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!currentFile) return;

        elements.uploadBtn.disabled = true;
        elements.uploadBtn.textContent = 'Uploading & Indexing...';
        elements.uploadStatus.hidden = true;

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/api/documents/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) throw new Error(data.detail || 'Upload failed');

            showStatus('PDF successfully uploaded and indexed!', 'success');
            setTimeout(() => {
                resetUpload();
                elements.uploadBtn.textContent = 'Upload & Index PDF';
                elements.uploadStatus.hidden = true;
                // Refresh the document list
                elements.docsLoading.hidden = false;
                elements.docsList.hidden = true;
                fetchDocuments();
            }, 2000);

        } catch (error) {
            console.error('Upload error:', error);
            showStatus(error.message, 'error');
            elements.uploadBtn.disabled = false;
            elements.uploadBtn.textContent = 'Upload & Index PDF';
        }
    });

    // Initial load
    fetchDocuments();
});
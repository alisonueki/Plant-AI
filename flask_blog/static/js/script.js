const dropArea = document.getElementById('drop-area');
const inputFile = document.getElementById('input-file');
const imageView = document.getElementById('img-view');

// Prevent default behavior for drag-and-drop events
inputFile.addEventListener("change", uploadImage);


function handleDrop(e) {
    let dt = e.dataTransfer;
    let files = dt.files;

    // Handle dropped files (you may want to add validation here)
    handleFiles(files);
}

function uploadImage(){
    let imgLink = URL.createObjectURL(inputFile.files[0]);
    console.log("Image selected:", imgLink);
    imageView.style.backgroundImage = `url(${imgLink})`;

}

dropArea.addEventListener("dragover", function(e){
    e.preventDefault();
});
dropArea.addEventListener("drop", function(e){
    e.preventDefault();
    inputFile.files = e.dataTransfer.files;
    uploadImage();
});
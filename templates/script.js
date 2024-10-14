var chatResponseBox = document.getElementById('chat-response-box');
var mainBg = document.getElementById('main-bg');
var userInput = document.getElementById('user-input');

$(document).ready(function () {
    // Tampilkan chat starter saat halaman dimuat
    chatResponseBox.classList.remove('d-none'); // Hapus kelas d-none
    chatResponseBox.innerHTML += createChatStarter(); // Tambahkan chat starter ke chatResponseBox

    $("#send-btn").click(async function (event) {
        event.preventDefault();
        const formData = new FormData();
        const userInputValue = document.getElementById('user-input').value;

        if (userInputValue == null || userInputValue == "") {
            Swal.fire({
                icon: 'error',
                title: 'Oops!!!',
                text: "Please enter some text!",
                allowOutsideClick: false,
                allowEscapeKey: false,
                confirmButtonColor: "#000"
            });
        } else {
            chatResponseBox.classList.remove('d-none');

            formData.append('prompt', userInputValue);

            var currentTimestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            var html = `
                <div class="row mb-5 chat-message user">
                    <div class="col-sm-3"></div>
                    <div class="col-sm-9 text-end">
                        <div class="row">
                            <div class="col-sm-11 message-content">
                                <h6 class="p-3 text-white mb-0">${userInputValue}</h6>
                                <p class="timestamp">${currentTimestamp}</p>
                            </div>
                            <div class="col-sm-1">
                                <img src="../static/assets/images/me-avatar.png" alt="Avatar" class="avatar">
                            </div>
                        </div>
                    </div>
                </div>

                <div id="loader" class="chat-message bot">
                    <div class="row">
                        <div class="col-sm-1">
                            <img src="../static/assets/images/bot-avatar.png" alt="Avatar" class="avatar">
                        </div>
                        <div class="col-sm-11 message-content">
                            <div class="snippet" data-title="dot-flashing">
                                <div class="stage">
                                    <div class="dot-flashing"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            chatResponseBox.innerHTML += html;
            document.getElementById('user-input').value = '';

            let response = await fetch('/chat_response', {
                method: "POST",
                body: formData
            });

            processChatResponse(response);
        }
    });

    userInput.addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
            event.preventDefault();
            document.getElementById("send-btn").click();
        }
    });
});

function createChatStarter() {
    var currentTimestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    var html = `
        <div class="row mb-5 chat-message bot"> 
            <div class="col-sm-1">
                <img src="../static/assets/images/bot-avatar.png" alt="Avatar" class="avatar">
            </div>
            <div class="col-sm-11 message-content">
                <h6 class="p-3 text-black mb-0">Hi, I'm XL Satu Bot! 👋 Got any question about XL Satu? I'd be happy to help!</h6>
                <p class="timestamp">${currentTimestamp}</p>
            </div>
        </div>
    `;
    return html;
}

async function processChatResponse(response){
    console.log('Processing response:', response); 
    switch (response.status) {
        case 400:
        case 404: // Handle 404 error (endpoint not found)
            Swal.fire({
                icon: 'error',
                title: 'Oops!!!',
                text: "Sorry, Couldn't be able to generate your response now. Please try after some time.",
                confirmButtonColor: "#040b14"
            });
            break;
        case 200:
            var json = await response.json();
            console.log('Parsed JSON:', json);
            var chatResult = json.answer.replace(/^(System|Assistant|Jawaban):\s*/, "");
            chatResult = chatResult.replace("\n", "<br>");
            console.log('Chat result:', chatResult);
            var currentTimestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            var html = `
                <div class="row mb-5 chat-message bot">
                    <div class="col-sm-1">
                        <img src="../static/assets/images/bot-avatar.png" alt="Avatar" class="avatar">
                    </div>
                    <div class="col-sm-11 message-content">
                        <h6 class="p-3 text-black mb-0">${chatResult}</h6>
                        <p class="timestamp">${currentTimestamp}</p>
                    </div>
                </div>
            `;

            var loader = document.getElementById('loader');
            loader.remove();
            chatResponseBox.innerHTML += html;
            break;
        default:
            Swal.fire({
                icon: 'error',
                title: 'Oops!!!',
                text: "There is a " + response.status + " error. Please contact admin for support.",
                confirmButtonColor: "#040b14"
            });
    }
}

var extractFilename = (path) => {
    const pathArray = path.split("/");
    const lastIndex = pathArray.length - 1;
    return pathArray[lastIndex];
};

var textarea_original_height = window.getComputedStyle(document.getElementById('question')).height;
var textarea_max_height = parseInt(window.getComputedStyle(document.getElementById('chat-messages')).height) * 0.8;
var chat_history = [];

function askQuestion() {
    var question = document.getElementById('question').value;
    var question_to_show = question.replace(/\n/g, "<br>");
    var responseHtml = "<div class='message user-message'><p>" + question_to_show + "</p></div>";
    document.getElementById("chat-messages").innerHTML += responseHtml;
    document.getElementById('question').value = ''; // Clear input field

    document.getElementById('send-question').disabled = true;


    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/ask", true);
    xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var response = JSON.parse(xhr.responseText);
            var answer = response.answer;
            var sources = response.sources;
            chat_history = response.chat_history;

            
            responseHtml = "<div class='message bot-message'><p>Respuesta: " + answer + "</p>";
            if (sources.length > 0) {
                responseHtml += "<p>Fuentes:</p>";
                sources.forEach(function (source) {
                    responseHtml += "<div class='source'>";
                            responseHtml += "<p class='source-source'>Fuente: " + source.source + "</p>";
                            responseHtml += "<p class='source-text'>Texto: " + source.text + "</p>";
                            responseHtml += "</div>";
                });
            }
            responseHtml += "</div>";

            document.getElementById("chat-messages").innerHTML += responseHtml;
            autoResize(); // Auto resize input field after clearing
            scrollToBottom();
            document.getElementById('send-question').disabled = false;
        }
    };
    xhr.send("question=" + question + "&chat_history=" + JSON.stringify(chat_history));
}

function more_than_one_line(textarea) {
    var lineHeight = parseInt(window.getComputedStyle(textarea).lineHeight);
    var lines_per_height = textarea.scrollHeight / lineHeight - 1;

    var lines_per_newline_characters = textarea.value.split('\n').length;
    return lines_per_height > 1 || lines_per_newline_characters > 1;
}

function check_max_height(textarea) {
    var height_textarea = parseInt(window.getComputedStyle(textarea).height);
    return height_textarea > textarea_max_height;
}

function autoResize() {
    var textarea = document.getElementById('question');
    if (textarea.value === '') {
        textarea.style.height = textarea_original_height;
    } else if (more_than_one_line(textarea)) {
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
        if (check_max_height(textarea)) {
            textarea.style.height = textarea_max_height + 'px';
            textarea.style.overflowY = 'scroll';
        }
    }
}

function scrollToBottom() {
    var chatMessages = document.getElementById('chat-messages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        if (event.shiftKey) {
            autoResize();
        } else {
            event.preventDefault();
            if (document.getElementById('send-question').disabled == false) {
                askQuestion();
            }
        }
    }

}
const inputField = document.getElementById('question');
inputField.addEventListener('keypress', handleKeyPress);
var username = 'admin'
var active_user = ''
var urlz = 'http://127.0.0.1:5000'

$(function () {
    var socket = io.connect(urlz)

    var default_font = "";
    
    var text_element = document.getElementById("input_text");
    text_element.focus();


    $("#input_text").on('keydown', function(e) {
        // e.preventDefault();
        if (e.which == 13) {
            console.log('Enter pressed!');
            document.getElementById("sendbtn").click();
            return false;
        }
    });

    $("#list").on('click', function(e) {
        var current_uname = e.target.childNodes[0].data;
        socket.emit('ask_for_convo', current_uname);
        active_user = current_uname;
        console.log(active_user)
        // console.log('List Item Click Event Triggered: ', e.target.childNodes[0].data === 'user1', e)
    })

    var add_text = function (uname, entered_text) {
        $("#inner_text").append(
            '<div class="row"> <div class="col"><b>' + uname + ': </b>' + entered_text + '</div> </div>')
        $("#input_text").val("")
        var objDiv = document.getElementById("inner");
        // objDiv.scrollTop = objDiv.scrollHeight;
    }

    var add_image = function (uname, images) {
        if (!images || images === [])
            return
        console.log(images)
        for (i = 0; i < images.length; i++) {
            $("#inner_text").append(
                '<div class="row"> <div class="col"><img src=' + '"' + images[i] + '"' + '></div> </div>')
        }
    }


    socket.on('connect', function () {
        console.log('Socket Connection Established!')
       
        socket.emit('uname', {
            data:'admin'
        })

        socket.on('connected_users', function(num_users) {
            for (var i = 0; i < parseInt(num_users); i++) {
                var ul = document.getElementById("list");
                var li = document.createElement("li");
                li.appendChild(document.createTextNode("group" + String(i)));
                ul.appendChild(li);
            }
        })

        socket.on('new_group', function(uname) {
            var ul = document.getElementById("list");
            var li = document.createElement("li");
            li.appendChild(document.createTextNode(uname));
            ul.appendChild(li);
        })

        socket.on('init_convo', function(conversation) {
            console.log('New Conversation Received:', conversation)
            const myNode = document.getElementById("inner_text");
            while (myNode.firstChild) {
                myNode.removeChild(myNode.firstChild);
            }

            for (ind = 0; ind < conversation.length; ind++) {
                add_text(conversation[ind].name, conversation[ind].text)
                add_image(conversation[ind].name, conversation[ind].images)
            }
            document.getElementById('inner').scrollTop = document.getElementById('inner').scrollHeight;
        })

        var form = $('form').on('submit', function(e) {
            e.preventDefault()
            var entered_text = $("#input_text").val()
            add_text(username, entered_text)
            socket.emit('admin_sendout', {
                'text': entered_text,
                'name': username,
                'active_user' : active_user
            })
            
            document.getElementById('inner').scrollTop = document.getElementById('inner').scrollHeight;
        })

    });

    socket.on('response', function(utterance) {
        console.log('This is the response for admin:', utterance)
        add_text(utterance.name, utterance.text)
        add_image(utterance.name, utterance.images)
        document.getElementById('inner').scrollTop = document.getElementById('inner').scrollHeight;
    })

    socket.on('add_score', function(score_json) {
        add_text('Judge', 'You obtained a score of ' + score_json['obtained'] + 
            'and now have a total score of ' + score_json['total'] + '.')
    })

    // $("#inner_text").append('<div class="row"> <div class="col text-left"><b>Bot: </b>Hi, I am Grit! Lets talk about your dataset!  </div> </div>')
})

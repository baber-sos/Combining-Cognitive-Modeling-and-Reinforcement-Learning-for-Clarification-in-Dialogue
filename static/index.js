var username = ''
//var urlz = 'http://ec2-3-21-104-223.us-east-2.compute.amazonaws.com'
var urlz = 'http://127.0.0.1:5000'

function start_client() {
    var socket = io.connect(urlz)
    
    var text_element = document.getElementById("input_text");
    //focus on the text box on loading of page
    text_element.focus();

    var add_text = function (uname, entered_text) {
        if (uname === "S") {
            $("#inner_text").append(
                '<div class="row"> <div class="col"><b>' + 'Director' + ': </b>' + entered_text + '</div> </div>')
        } else if (uname === "L"){
            $("#inner_text").append(
                '<div class="row right"> <div class="col"><b>' + 'Matcher' + ': </b>' + entered_text + '</div> </div>')
        } else if (uname === "State Information"){
            $("#inner_text").append(
                '<div class="row"> <div class="col"><b>' + uname + ': </b>' + '<span  style="color:blue">' + entered_text + '</span></div> </div>')
        } else {
            $("#inner_text").append(
                '<div class="row right"> <div class="col"><b>' + uname + ': </b>' + entered_text + '</div> </div>')
        }
        $("#input_text").val("")
        var objDiv = document.getElementById("inner");
        objDiv.scrollTop = objDiv.scrollHeight;
    }

    socket.on('connect', function () {
        console.log('Socket Connection Established!')
        
        socket.emit('uname', {
            data:'connection made'
        })

        socket.on('uname', function(clr_json) {
            console.log('recvd uname: ', clr_json)
            username = clr_json['name']

            //clear the previous chat
            var e = document.getElementById('inner_text')
            var child = document.getElementById('inner_text').lastElementChild;  
            while (child) { 
                e.removeChild(child); 
                child = e.lastElementChild; 
            }

            var i = 0
            var j = 0
            var ele;
            //add text on top of the boxes
            // for (i = 0; i < clr_json['str'].length; i++){
            //     ele = document.getElementById('f' + parseInt(i + 1))
            //     ele.append(document.createTextNode(clr_json['str'][i]))
            // }
            var form_ele = document.getElementsByTagName('form')[0]
            form_ele.hidden = false
            //add colors to the boxes
            for (var i = 0; i < clr_json['colors'].length; i++){
                ele = document.getElementById('box' + parseInt(i + 1))
                var this_color = '#'
                
                for (var j = 0; j < 3; j++){
                    var clr_cnst = clr_json['colors'][i][j].toString(16)
                    if (clr_cnst.length < 2)
                        clr_cnst = '0' + clr_cnst
                    this_color += clr_cnst
                }

                console.log(this_color.length, this_color)
                if (username === "S" && i === 0) {
                    ele.style.backgroundImage = 'linear-gradient('
                        + '180deg, black, black 10%,' + this_color + ' 5%,' +
                        this_color + ' 90%,' + 'black 5%';
                } else
                    ele.style.backgroundColor = this_color
                
                loader_elem = document.getElementById('loading_screen')
                loader_elem.hidden = true

                rest_ele = document.getElementById('chat')
                action_ele = document.getElementById('action_buttons')
                box_ele = document.getElementById('color_boxes')
                rest_ele.hidden = false
                action_ele.hidden = false
                box_ele.hidden = false
            }
	    if (clr_json.length > 0 && (clr_json['str'][0].length) > 0)
	        add_text('System', clr_json['str'][0])
        })

        socket.on('init_convo', function(conversation) {
            console.log(conversation)
            for (ind = 0; ind < conversation.length; ind++) {
                add_text(conversation[ind].name, conversation[ind].question)
            }
            document.getElementById('inner').scrollTop = document.getElementById('inner').scrollHeight;
        })

        socket.on('task_over', function(msg){
            loader_elem = document.getElementById('loading_screen')
            loader_elem.hidden = true

            rest_ele = document.getElementById('chat')
            action_ele = document.getElementById('action_buttons')
            box_ele = document.getElementById('color_boxes')
            rest_ele.hidden = false
            action_ele.hidden = false
            box_ele.hidden = false

            add_text(msg.name, msg.text)
            document.getElementById('inner').scrollTop = document.getElementById('inner').scrollHeight;
            element = document.getElementById('next_button')
            element.style.display = "none";

            document.getElementById('input_text').hidden = true
            document.getElementById('sendbtn').hidden = true
        })

        var form = $('form').on('submit', function(e) {
            e.preventDefault()
            var cele = document.getElementById('code_text');
            var entered_text = $("#input_text").val()
            if (entered_text.length == 0) {
                add_text('System', 'Please provide a color description.')
            } else {
                add_text(username, entered_text)
                // add_text('L', 'Thinking.....')
                console.log('This is the entered text', entered_text)
                socket.emit('sendout', {'name' : username, 'text' : entered_text})
                document.getElementById('inner').scrollTop = document.getElementById('inner').scrollHeight;
            }
        })

        document.getElementById('reset_button').onclick = function() {
            console.log('Reset Event Encountered!')
            socket.emit('reset', {})
            e = document.getElementById('inner_text')
            var child = e.lastElementChild;  
            while (child) { 
                e.removeChild(child); 
                child = e.lastElementChild; 
            } 
        }

        document.getElementById('next_button').onclick = function() {
            console.log('Next Event Encountered!')
            socket.emit('next', {})

            e = document.getElementById('inner_text')
            loader_elem = document.getElementById('loading_screen')
            loader_elem.hidden = false

            rest_ele = document.getElementById('chat')
            action_ele = document.getElementById('action_buttons')
            box_ele = document.getElementById('color_boxes')
            rest_ele.hidden = true
            action_ele.hidden = true
            box_ele.hidden = true
        }

    });

    socket.on('response', function(utterance) {
        console.log('This is the response for user:', utterance)
        if (document.getElementById('loading_screen').hidden === false) {
            document.getElementById('loading_screen').hidden = true
            document.getElementById('chat').hidden = false
            document.getElementById('action_buttons').hidden = false
            document.getElementById('color_boxes').hidden = false
        }
        if (utterance.info != '')
            add_text('State Information', utterance.info)
            
        add_text(utterance.name, utterance.text)
        
        document.getElementById('inner').scrollTop = document.getElementById('inner').scrollHeight;
    })

    socket.on('add_score', function(score_json) {
        if (score_json.info !== '')
            add_text('State Information', score_json.info)
        var form_ele = document.getElementsByTagName('form')[0]
        form_ele.hidden = true
        if (score_json.result === "passed")
            add_text('System Status', 'You have successfully completed this round.')
        else
            add_text('System Status', 'I am sorry but matcher failed to choose the right color patch.')
    })

    $('#color_boxes').on('click', function (event) {
        console.log(event.target)
        console.log(event.target.style.backgroundColor)
        if (username !== 'S')
            socket.emit('score', event.target.style.backgroundColor)
    })
    
}

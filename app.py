import sys
#sys.path.insert(0, '/home/ubuntu/Dialogue_Research/color_in_context/system/dialogue_manager')
sys.path.insert(0, '/home/sos/CIC/system/dialogue_manager')
import json as js
from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO, emit, disconnect
import random
from datetime import datetime
from task_manager.task_manager import task_manager
import os

from generation.gen_rules import check_social_stuff
import string

import logging
logging.basicConfig(filename='./dialogue_manager/logs/error.log',level=logging.DEBUG)

global session_count;
session_count = 0;

active_userind = -1;

people_per_room = 1
sessions = list()   #contains groups of 3 people per room
score = list()
session_ix_map = dict()  #contains index of group in 'sessions' for each session
conversations = dict()  #maps group index to conversation happening
colors = list() #(-1, []);
task_counts = list()

admin_sid = str();

conv_manager = task_manager()

flag = False;

application = Flask(__name__,static_folder='static')
app = application
socketio = SocketIO(app, async_mode='gevent')

@app.route('/', methods=['GET', 'POST'])
def index():
    global session_count
    # print("***************index function was called", session_count % people_per_room)  
    return send_from_directory(app.static_folder, 'index.html', cache_timeout=0)

@app.route('/static/<file_name>')
def static_file(file_name):
    return send_from_directory(app.static_folder, file_name)

@app.route('/favicon.ico')
def icon():
    return '';


@app.route('/admin', methods=['GET', 'POST'])
def admin_index():
    print("index function was called")
    return send_from_directory(app.static_folder, 'admin_index.html')

@app.route('/admin_static/<file_name>')
def admin_static_file(file_name):
    return send_from_directory(app.static_folder, file_name)

@app.route('/pics/<image_name>')
def get_image(image_name):
    print('Asked for an image!', image_name)
    return send_from_directory('./pics/', image_name)

@socketio.on('uname')
def handle_connection(json):
    # print('**************Session ID:', request.sid)
    global session_count
    global admin_sid

    if json['data'] == 'admin':
        emit('connected_users', len(sessions))
        admin_sid = request.sid
    else:
        colors_to_send = []
        client_name = ''
        str_to_send = []
        cts = []
        if sessions == [] or len(sessions[-1]) == people_per_room:
            #this is the first person to join so we'll send this one the name sequence
            #and the colors
            cur_colors, condition, csv_ix = conv_manager.sample_color()
            print('Received Colors')

            sessions.append([request.sid])
            conversations.setdefault(len(sessions) - 1, [])
            conversations[len(sessions) - 1].append({'condition' : condition,
                                                    'row_index' : int(csv_ix)})
            score.append(0)
            task_counts.append(0)

            colors.append(cur_colors)
            client_name = 'S'
            #str_to_send = ['Please Enter your Mturk Id.']
            #if os.getenv('MTURK') == 'True':
            #    emit('response', {'name' : 'System',
            #        'info': '',
            #        'text' : 'Please Enter your Mturk Id.'}, room=request.sid)
            print('New Session Encountered:', len(sessions))
            
        elif len(sessions[-1]) < people_per_room:
            sessions[-1].append(request.sid)
            client_name = 'L'
            cts = [c for c in colors[-1]]
            random.shuffle(cts)
        
        session_ix_map[request.sid] = len(sessions) - 1


        emit('uname', {'name' : client_name,
                        'colors': (colors[-1] if client_name == 'S' else cts),
                        'str' : str_to_send})

        if admin_sid != '' and len(sessions[-1]) == 1:
            emit('new_group', 'group' + str(len(sessions) - 1), room=admin_sid)
        session_count += 1
        

def msg_recvd():
    print("I got the message");

flag = True

def check_alphanum(text):
    uppercase = string.ascii_uppercase
    for char in text:
        if not (char in uppercase or char.isdigit()):
            return False
    return True

def check_if_mturk_id(text):
    if len(text) >= 12 and check_alphanum(text):
        return True
    return False

@socketio.on('sendout')
def inputoutput(json):
    global flag
    group_index = session_ix_map[request.sid]
    print('@@@@@@@@@@@@@@@')
    print('THis is the received JSON:', json, conversations[group_index][-1])
    print('@@@@@@@@@@@@@@@')
    if 'success' in conversations[group_index][-1]:
        return
    json['timestamp'] = str(datetime.now())
    json['finished'] = 0
    conversations[group_index].append(json)

    if check_if_mturk_id(json['text']):
        emit('response', {'text' : 'Thank you for submitting your MturkID.' + \
            ' Now please provide me with the target description.',
            'name' : 'System',
            'info' : ''}, room=sessions[group_index][0])
        conversations[group_index].append({'text' : 'Thank you for submitting your MturkID.' + \
            ' Now please provide me with the target description.',
            'name' : 'System',
            'info' : ''})
        return
    print('@@@@@@@@@@@@@@@')
    print('THis is the received JSON:', json)
    print('@@@@@@@@@@@@@@@')
    social_response = check_social_stuff(json['text'])
    if len(social_response) != 0:
        conversations[group_index].append({'text' : social_response,
                'name' : 'Matcher',
                'info' : '',
                'timestamp' : str(datetime.now())
            })

        emit('response', {'text' : social_response,
                'name' : 'Matcher',
                'info' : ''}, room=sessions[group_index][0])
        return


    # if os.getenv('MTURK') == 'True' and task_counts[group_index] < 1:
    #     if len(conversations[group_index]) <= 2:
    #         emit('response', {'text' : 'Thank you. You can start your task now.',
    #             'name' : 'System',
    #             'info' : ''}, room=sessions[group_index][0])
    #         return

    #send this conversation json to the other people in the conversation
    finish_flag = False
    if people_per_room == 1:
        #conversing with a computer agent
        if len(json['text']) == 0:
            emit('response', {'name' : 'System',\
                'text' : 'Please provide with a color description.'}, room=sessions[group_index][0])
            return
        response, action = conv_manager.get_response(json, group_index)
        if response == None and action == None:
            response = {'name' : 'System', \
                'text' : 'This task has exceed maximum length. Please proceed to the next task (if any).'}
            
            emit('response', response, room=sessions[group_index][0])
            finish_flag = True
            response['success'] = 0
        elif action != None:
            # response = dict()
            if action == True:
                emit('add_score', {'result' : 'passed', 'info' : response['info']}, \
                    room=sessions[group_index][0])
                response['success'] = 1
            else:
                emit('add_score', {'result' : 'failed', 'info' : response['info']}, \
                    room=sessions[group_index][0])
                response['success'] = 0
            finish_flag = True
        else:
            emit('response', response, room=sessions[group_index][0])
        response['timestamp'] = str(datetime.now())
        response['finished'] = int(finish_flag)
        conversations[group_index].append(response)
        if action != None:
            with open('./conversations/' + str(request.sid) + '_' +\
                str(task_counts[group_index]) + '.json', 'a') as conv_file:

                js.dump(conversations[group_index], conv_file, indent=4)
                conversations[group_index] = []
    else:
        for person in sessions[group_index]:
            if person != request.sid:
                emit('response', json, room=person)

    if active_userind != -1 and request.sid in sessions[active_userind]:
        emit('response', json, room=admin_sid)

@socketio.on('admin_sendout')
def admin_inputoutput(json):
    print('&&&&&&&&&&&&&&&&&&&&&&&Admin Sendout event received with following json:', json)

    for person in sessions[active_userind]:
        emit('response', json, room=person)
    conversations[active_userind].append(json)

@socketio.on('ask_for_convo')
def ask_convo(uname):
    ix = int(uname[5:])
    global active_userind
    active_userind = ix
    emit('init_convo', conversations[ix])

@socketio.on('disconnect')
def handle_close():
    print('socket closed')
    if request.sid != admin_sid:
        group_index = session_ix_map[request.sid]
        if len(conversations[group_index]) == 0:
            return
        with open('./conversations/' + str(request.sid) + '_' +\
            str(task_counts[group_index]) + '.json', 'w') as conv_file:
        # with open('./conversations/' + str(request.sid) + '.json', 'w') as conv_file:
            js.dump(conversations[group_index], conv_file, indent=4)

@socketio.on('score')
def receive_score(judged_score):
    ix = session_ix_map[request.sid]
    if request.sid == sessions[ix][0]:
        return
    else:
        recv_color = tuple(int(x) for x in judged_score.split('(')[1].split(')')[0].split(','))
        if recv_color == colors[ix][0]:
            for person in sessions[ix]:
                emit('add_score', 'passed', room=person)
        else:
            for person in sessions[ix]:
                emit('add_score', 'failed', room=person)

@socketio.on('reset')
def reset(event):
    group_index = session_ix_map[request.sid]
    with open('./conversations/' + str(request.sid) + '_' + \
        str(task_counts[group_index]) + '.json', 'a') as conv_file:

        js.dump(conversations[group_index], conv_file, indent=4)
        conversations[group_index] = []
        task_counts[group_index] += 1
    conv_manager.reset(group_index)

@socketio.on('next')
def next(event):
    group_index = session_ix_map[request.sid]

    task_counts[group_index] += 1

    cur_colors, condition, csv_ix = conv_manager.get_next(group_index)
    print("After the next event: ", cur_colors)
    if len(cur_colors) == 0:
        emit('response', {'name' : 'System', \
                        'text' : 'Please finish the task before proceeding.'})
        return
    elif cur_colors == 'None':
        print(' I AM HERE ', cur_colors)
        emit('task_over', {'name': 'System Status',
                        'text' : 'The task is over. Thank you for your participation. '})
        # disconnect()
        conversations[group_index] == []
        return

    conversations[group_index].append({'condition' : condition,
                                        'row_index' : int(csv_ix)})

    colors[group_index] = cur_colors
    emit('uname', {'name' : 'S',
                    'colors': cur_colors,
                    'str' : []})
    
        



socketio.run(app, host='0.0.0.0', port=5000, debug=True)
#socketio.run(app, host='172.31.10.8', port=80, debug=True)




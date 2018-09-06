const {remote} = require('electron');
const {dialog, session} = remote;

$(document).ready(function() {
    session.defaultSession.cookies.get({url: 'http://localhost:5000'}, (error, cookies) => {
        if(error) console.error(error);
        if(cookies.length < 3) return;
        const dataDir = cookies.filter(cookie => cookie['name'] == 'dataDir')[0]['value'];
        const saveDir = cookies.filter(cookie => cookie['name'] == 'saveDir')[0]['value'];
        const latentSize = cookies.filter(cookie => cookie['name'] == 'latentSize')[0]['value'];
        $('#dataDirectory').val(dataDir);
        $('#saveDirectory').val(saveDir);
        $('#latentSize').val(latentSize);
    });

    function trySkipInit() {
        $('#loadingScreen').removeClass('hidden');
        $.get('http://localhost:5000/init', function(data) {
            if(data == 'success') {
                remote.getCurrentWindow().loadURL(`file://${__dirname}/eval.html`)
            }
            $('#loadingScreen').addClass('hidden');
        });
    }

    $('#selectDataDirectory').click(function() {
        dialog.showOpenDialog({ properties: ['openDirectory'] }, function(filename) {
            $('#dataDirectory').val(filename);
        });
    });

    $('#selectSaveDirectory').click(function() {
        dialog.showOpenDialog({ properties: ['openDirectory'] }, function(filename) {
            $('#saveDirectory').val(filename);
        });
    });

    $('#properties').submit(function(e) {
        e.preventDefault();
        $('#err').text('');
        $('#loadingScreen').removeClass('hidden');
        const url = 'http://localhost:5000/init?datapath=' + $('#dataDirectory').val() +
                    '&savepath=' + $('#saveDirectory').val() + '&latentsize=' + $('#latentSize').val();
        $.get(url, function(data) {
            $('#loadingScreen').addClass('hidden');
            if(data == 'success') {
                const dataDir = {url: 'http://localhost:5000', name: 'dataDir', value: $('#dataDirectory').val()};
                const saveDir = {url: 'http://localhost:5000', name: 'saveDir', value: $('#saveDirectory').val()};
                const latentSize = {url: 'http://localhost:5000', name: 'latentSize', value: $('#latentSize').val()};
                $.each([dataDir, saveDir, latentSize], function(i, cookie) {
                    session.defaultSession.cookies.set(cookie, (error) => {
                        if(error) console.error(error);
                    });
                })
                remote.getCurrentWindow().loadURL(`file://${__dirname}/eval.html`)
            } else {
                $('#err').text(data);
            }
        });
    });

    trySkipInit();
});

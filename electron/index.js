const {remote} = require('electron');
const {dialog} = remote;

$(document).ready(function() {
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
                remote.getCurrentWindow().loadURL(`file://${__dirname}/eval.html`)
            } else {
                $('#err').text(data);
            }
        });
    });

    trySkipInit();
});

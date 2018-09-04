const {remote} = require('electron');
const {dialog} = remote;

$(document).ready(function() {
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
});

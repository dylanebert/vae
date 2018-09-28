const {remote} = require('electron')
const {dialog, Menu, MenuItem} = remote
const fs = remote.require('fs')
const path = remote.require('path')

//Menu
var config = null

function reload() {
    $('.dropdown-item').click(function() {
        var label = $(this).text()
        $.get('http://localhost:5000/image?label=' + label, function(data) {
            $('#title').text(label)
            $('#reconstruction').attr('src', 'data:image/jpeg;base64, ' + data)
        })
    })
}

function populateClasses() {
    $.get('http://localhost:5000/classes', function(json) {
        var data = $.parseJSON(json)
        $('.dropdown-item').remove()
        $('#helpOpen').remove()
        $(data).each(function(i, entry) {
            var html = '<li><a class="dropdown-item" href="#">' + entry + '</a></li>'
            $('#dropdown').append(html)
        })
        reload()
        $('#loadingScreen').css('display', 'none')
    })
}

function loadModel() {
    dialog.showOpenDialog({ filters: [{ name: 'JSON', extensions: ['json'] }], properties: ['openFile'] }, function(filenames) {
        if(filenames == undefined) {
            console.log('No file selected')
            return
        }

        var filename = filenames[0]
        $('#loadingScreen').css('display', '')
        $('#loadingText').text('Loading model')
        $.get('http://localhost:5000/load?path=' + filename, function(data) {
            if(data == '1') {
                populateClasses()
            } else {
                console.log(data)
                $('#loadingScreen').css('display', 'none')
            }
        })
    })
}

const template = [{
    label: 'File',
    submenu: [{
        label: 'Load Model', click() {
            loadModel()
        }
    }]
}]

const menu = Menu.buildFromTemplate(template)
Menu.setApplicationMenu(menu)

$('#helpOpen').click(loadModel)

//Dropdown
$('#dropdownSearch').keyup(function() {
    const filter = $(this).val().toUpperCase()
    $('.dropdown-item').each(function(i, elem) {
        if($(elem).text().toUpperCase().indexOf(filter) > -1) {
            $(elem).css('display', '')
        } else {
            $(elem).css('display', 'none')
        }
    })
})

$(document).ready(function() {
    $('#loadingScreen').css('display', 'none')
    $.get('http://localhost:5000/is-loaded', function(data) {
        if(data == '1') {
            populateClasses()
        }
    })
})

const {remote} = require('electron')
const {dialog, Menu, MenuItem} = remote
const fs = remote.require('fs')
const path = remote.require('path')

//Menu
var config = null

function reload() {
    $('.dropdown-item-c1').click(function() {
        var label = $(this).text()
        $('#loadingScreen').css('display', '')
        $.get('http://localhost:5000/image?label=' + label, function(data) {
            $('#c1-title').text(label)
            $('#c1-reconstruction').attr('src', 'data:image/jpeg;base64, ' + data)
            $('#loadingScreen').css('display', 'none')
            $('.right').css('display', '')
        })
    })
    $('.dropdown-item-c2').click(function() {
        var label = $(this).text()
        $('#loadingScreen').css('display', '')
        $.get('http://localhost:5000/image?label=' + label, function(data) {
            $('#c2-title').text(label)
            $('#c2-reconstruction').attr('src', 'data:image/jpeg;base64, ' + data)
            $('#loadingScreen').css('display', 'none')
        })
    })
}

function populateClasses() {
    $.get('http://localhost:5000/classes', function(json) {
        var data = $.parseJSON(json)
        $('.dropdown-item-c1').remove()
        $('.helpOpen').remove()
        $(data).each(function(i, entry) {
            var html1 = '<li><a class="dropdown-item-c1" href="#">' + entry + '</a></li>'
            var html2 = '<li><a class="dropdown-item-c2" href="#">' + entry + '</a></li>'
            $('#dropdown-c1').append(html1)
            $('#dropdown-c2').append(html2)
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

$('.helpOpen').click(loadModel)

//Dropdown
$('#c1-search').keyup(function() {
    const filter = $(this).val().toUpperCase()
    $('.dropdown-item-c1').each(function(i, elem) {
        if($(elem).text().toUpperCase().indexOf(filter) > -1) {
            $(elem).css('display', '')
        } else {
            $(elem).css('display', 'none')
        }
    })
})

$(document).ready(function() {
    $('#loadingScreen').css('display', 'none')
    $('.right').css('display', 'none')
    $.get('http://localhost:5000/is-loaded', function(data) {
        if(data == '1') {
            populateClasses()
        }
    })
})

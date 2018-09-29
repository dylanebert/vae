const {remote} = require('electron')
const {dialog, Menu, MenuItem} = remote
const fs = remote.require('fs')
const path = remote.require('path')

const encodingColors = {'c1': 'rgba(225, 0, 0, .2)', 'c2': 'rgba(0, 0, 225, .2)'}

var config = null
var ctx = $('#plot')[0].getContext('2d')
var chart = new Chart(ctx, {
    type: 'scatter',
    data: {
        datasets: []
    },
    options: {
        scales: {
            xAxes: [{
                type: 'linear',
                position: 'bottom',
                ticks: {
                    max: 4,
                    min: -4
                }
            }],
            yAxes: [{
                ticks: {
                    max: 4,
                    min: -4
                }
            }]
        }
    }
})

function reload() {
    $('.dropdown-item').click(function() {
        var label = $(this).text()
        var c = ''
        if($(this).hasClass('c1')) {
            c = 'c1'
            $('#c2-title').text('')
            $('#c2-reconstruction').attr('src', '')
        } else {
            c = 'c2'
        }
        $('#loadingScreen').css('display', '')
        $('#loadingText').text('Loading class data')
        $.get('http://localhost:5000/data?label=' + label, function(data) {
            var parsed = $.parseJSON(data)
            var img = parsed.img
            var mean = parsed.mean
            var encodings = parsed.encodings
            $('#' + c + '-title').text(label)
            $('#' + c + '-reconstruction').attr('src', 'data:image/jpeg;base64, ' + img)
            $('#loadingScreen').css('display', 'none')
            if(c == 'c1') {
                while(chart.data.datasets.length > 0) {
                    chart.data.datasets.pop()
                }
            } else {
                if(chart.data.datasets.length > 1)
                    chart.data.datasets.pop()
            }
            chart.data.datasets.push({label: label, data: encodings, backgroundColor: encodingColors[c]})
            chart.update()
        })
    })
}

function populateClasses() {
    $.get('http://localhost:5000/classes', function(json) {
        var data = $.parseJSON(json)
        $('.dropdown-item').remove()
        $('.helpOpen').remove()
        $(data).each(function(i, entry) {
            var html1 = '<li><a class="dropdown-item c1" href="#">' + entry + '</a></li>'
            var html2 = '<li><a class="dropdown-item c2" href="#">' + entry + '</a></li>'
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
$('.dropdown-search').keyup(function() {
    const filter = $(this).val().toUpperCase()
    $(this).parent().parent().find('.dropdown-item').each(function(i, elem) {
        if($(elem).text().toUpperCase().indexOf(filter) > -1) {
            $(elem).css('display', '')
        } else {
            $(elem).css('display', 'none')
        }
    })
})

$('.dropdown').on('hide.bs.dropdown', function() {
    $('.dropdown-search').val('')
    $('.dropdown-search').trigger('keyup')
})

$(document).ready(function() {
    $('#loadingScreen').css('display', 'none')
    $.get('http://localhost:5000/is-loaded', function(data) {
        if(data == '1') {
            populateClasses()
        }
    })
})

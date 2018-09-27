const {remote} = require('electron')
const {dialog, Menu, MenuItem} = remote

//Menu
var config = null

function reload() {
    $('.dropdown-item').click(function() {
        console.log($(this).text())
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
    })
}

function loadModel() {
    dialog.showOpenDialog({ filters: [{ name: 'JSON', extensions: ['json'] }], properties: ['openFile'] }, function(filenames) {
        if(filenames == undefined) {
            console.log('No file selected')
            return
        }

        var filename = filenames[0]
        $.get('http://localhost:5000/load?path=' + filename, function(data) {
            if(data == '1') {
                populateClasses()
            } else {
                console.log(data)
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

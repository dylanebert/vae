const {remote} = require('electron')
const {dialog, Menu, MenuItem} = remote
const fs = require('fs')

var config = null

function populate(class_names) {

}

const template = [{
    label: 'File',
    submenu: [{
        label: 'Load Model', click() {
            dialog.showOpenDialog({ filters: [{ name: 'JSON', extensions: ['json'] }], properties: ['openFile'] }, function(filenames) {
                if(filenames == undefined) {
                    console.log('No file selected')
                    return
                }

                var filename = filenames[0]
                fs.readFile(filename, 'utf-8', (err, data) => {
                    if(err) {
                        console.log(err.message)
                        return
                    }
                    config = $.parseJSON(data)
                    fs.readFile(config.means_path, 'utf-8', (err, data) => {
                        if(err) {
                            console.log(err.message)
                            return
                        }
                        var means = $.parseJSON(data)
                        console.log(means)
                    })
                })
            })
        }
    }]
}]

const menu = Menu.buildFromTemplate(template)
Menu.setApplicationMenu(menu)

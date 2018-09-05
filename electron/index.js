const electron = require('electron');
const app = electron.app;
const BrowserWindow = electron.BrowserWindow;

require('electron-reload')(__dirname)

let mainWindow;

function createWindow() {
	mainWindow = new BrowserWindow({minWidth: 1200, minHeight: 800, width: 1200, height: 800});
	mainWindow.setMenu(null);
	mainWindow.webContents.openDevTools();
	mainWindow.loadURL(`file://${__dirname}/index.html`);

	mainWindow.on('closed', function() {
		mainWindow = null;
	});
}

app.on('ready', createWindow);

app.on('window-all-closed', function() {
	if(process.platform !== 'darwin') {
		app.quit();
	}
});

app.on('activate', function() {
	if(mainWindow == null) {
		createWindow();
	}
});

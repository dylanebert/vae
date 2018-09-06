const {remote} = require('electron');

$(document).ready(function() {
    function initialize() {
        $('.classItem').click(function() {
            if($(this).hasClass('active')) {
                $(this).removeClass('active');
            } else {
                $.get('http://localhost:5000/get?class=' + $(this).text(), function(data) {
                    console.log(data)
                    $('<img>', {
                        'src': 'data:image/png;base64,' + data,
                        'width': '64px', 'height': '64px'
                    }).appendTo($('#view'));
                });
                $(this).addClass('active');
            }
        });
    }

    $.get('http://localhost:5000/classes', function(data) {
        const classes = $.parseJSON(data);
        $.each(classes, function(i, entry) {
            $('#classList').append('<li class="list-group-item classItem">' + entry + '</li>');
        });
        initialize();
    });

    $('#backButton').click(function() {
        remote.getCurrentWindow().loadURL(`file://${__dirname}/index.html`)
    });

    $('#classSearch').keyup(function() {
        var filter = $('#classSearch').val().toUpperCase();

        $('ul li').each(function(i, item) {
            if($(item).text().toUpperCase().indexOf(filter) > -1) {
                $(item).removeClass('hidden');
            } else {
                $(item).addClass('hidden');
            }
        });
    });
});

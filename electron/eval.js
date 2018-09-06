const {remote} = require('electron');

$(document).ready(function() {
    function initialize() {
        $('.classItem').click(function() {
            var classItem = $(this);
            if($(this).hasClass('active')) return;
            $('.backdrop').removeClass('hidden');
            $('#imgView').html('');
            $.get('http://localhost:5000/get?class=' + classItem.text(), function(data) {
                $('<img>', {
                    'src': data,
                    'width': '512px', 'height': '512px'
                }).appendTo($('#imgView'));
                $('.backdrop').addClass('hidden');
                $('.active').removeClass('active');
                $(classItem).addClass('active');
            });
        });
        $('#classList li').first().trigger('click');
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

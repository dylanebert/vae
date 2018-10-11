const {remote} = require('electron')
const {dialog, Menu, MenuItem} = remote
const fs = remote.require('fs')
const path = remote.require('path')

const meanColors = {'c1': 'rgba(225, 0, 0, 1)', 'c2': 'rgba(0, 0, 225, 1)'}
const encodingColors = {'c1': 'rgba(225, 0, 0, .2)', 'c2': 'rgba(0, 0, 225, .2)'}

const red = 'rgba(255, 94, 87,1.0)'
const yellow = 'rgba(255, 221, 89,1.0)'
const green = 'rgba(11, 232, 129,1.0)'

var active = {'c1': null, 'c2': null}

var config = null
var ctx = $('#plot')[0].getContext('2d')
var chartData = {'c1': null, 'c2': null}
var chart = new Chart(ctx, {
    type: 'scatter',
    data: {
        datasets: []
    },
    options: {
        tooltips: {
                callbacks: {
                    label: function(tooltipItem, data) {
                        var label = data.datasets[tooltipItem.datasetIndex].label || ''
                        if(label) {
                            label += ': '
                        }
                        label += Math.round(tooltipItem.xLabel * 100) / 100
                        label += ', '
                        label += Math.round(tooltipItem.yLabel * 100) / 100
                        return label
                    }
                }
        }/*,
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
        }*/
    }
})

function classClick(label, c) {
    $('#loadingScreen').css('display', '')
    $('#loadingText').text('Loading class data')
    active[c] = label
    $.get('http://localhost:5000/data?label=' + label, function(data) {
        var parsed = $.parseJSON(data)
        var mean_encoding = parsed.mean
        var encodings = parsed.encodings

        //Populate sidebar
        $('#' + c + '-reconstruction-mean').attr('src', 'data:image/jpeg;base64, ' + parsed.img_mean)
        $('#' + c + '-reconstruction-nearest').attr('src', 'data:image/jpeg;base64, ' + parsed.img_nearest)
        $('#' + c + '-reconstruction-random').attr('src', 'data:image/jpeg;base64, ' + parsed.img_random)
        $('#loadingScreen').css('display', 'none')

        //Populate accuracy table
        $('#' + c + '-table-title').text('"' + label + '" test accuracy:')
        $.each(['r1', 'r5', 'r10', 'r25', 'r50'], function(i, r) {
            $.each(['euc', 'cos', 'an'], function(j, s) {
                var idx = r + '-' + s
                var rVal = Math.round(parsed[idx] * 100) / 100
                var bgColor = red
                if(rVal >= .25) {
                    bgColor = yellow
                }
                if(rVal >= .75) {
                    bgColor = green
                }
                $('#' + c + '-' + idx).text(rVal).css('background-color', bgColor)
            })
        })
        $.each(['p5', 'p75', 'p9'], function(i, p) {
            var pVal = Math.round(parsed[p] * 100) / 100
            var bgColor = red
            if(pVal >= .25) {
                bgColor = yellow
            }
            if(pVal >= .75) {
                bgColor = green
            }
            $('#' + c + '-' + p).text(pVal).css('background-color', bgColor)
        })

        //Populate chart
        var chartDatasets = []
        var bgColors = []
        $.each(encodings, function(e) { bgColors.push(encodingColors[c]); })
        encodings.push(mean_encoding)
        bgColors.push(meanColors[c])
        chartData[c] = {label: label, data: encodings, backgroundColor: encodingColors[c], pointBackgroundColor: bgColors}
        if(chartData['c1'] != null) {
            chartDatasets.push(chartData['c1'])
        }
        if(chartData['c2'] != null) {
            chartDatasets.push(chartData['c2'])
        }
        chart.data.datasets = chartDatasets
        chart.update()
    })

    //Populate entailment
    if (active['c1'] != null && active['c2'] != null) {
        var html_str = active['c1'] + ' <span class="glyphicon glyphicon-arrow-right"></span> ' + active['c2']
        $('#en-table-title').html(html_str)
        $.get('http://localhost:5000/entailment?c1=' + active['c1'] + '&c2=' + active['c2'], function(data) {
            var parsed = $.parseJSON(data)
            $.each(['r1', 'r5', 'r10', 'r25', 'r50'], function(i, r) {
                $.each(['euc', 'cos', 'an'], function(j, s) {
                    var idx = r + '-' + s
                    var rVal = Math.round(parsed[idx] * 100) / 100
                    var bgColor = red
                    if(rVal >= .25) {
                        bgColor = yellow
                    }
                    if(rVal >= .75) {
                        bgColor = green
                    }
                    $('#en-' + idx).text(rVal).css('background-color', bgColor)
                })
            })
            $.each(['p5', 'p75', 'p9'], function(i, p) {
                var pVal = Math.round(parsed[p] * 100) / 100
                var bgColor = red
                if(pVal >= .25) {
                    bgColor = yellow
                }
                if(pVal >= .75) {
                    bgColor = green
                }
                $('#en-' + p).text(pVal).css('background-color', bgColor)
            })
        })
    }
}

function populateClasses() {
    $.get('http://localhost:5000/classes', function(json) {
        var data = $.parseJSON(json)
        $.each(['c1', 'c2'], function(i, c) {
            $('#' + c + '-autocomplete').autoComplete({
                minChars: 1,
                source: function(term, suggest) {
                    term = term.toLowerCase()
                    var choices = data
                    var suggestions = []
                    for (i = 0; i < choices.length; i++)
                        if (~choices[i].toLowerCase().indexOf(term)) suggestions.push(choices[i])
                    suggest(suggestions)
                },
                onSelect: function(e, term, item) {
                    classClick(term, c)
                }
            })
        })
        $('#loadingScreen').css('display', 'none')
    })
}

//Check if model already loaded on server
$(document).ready(function() {
    populateClasses()
})

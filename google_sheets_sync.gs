/**
 * Model history -> Google Sheets.
 *
 * Pulls the pick history that run_weekly.bat publishes to GitHub Pages and keeps one tab per week
 * ("NFL 2026 Wk 4", "College 2026 Wk 5", ...) plus a "Summary" tab, in this sheet.
 *
 * One-time setup (see README "Google Sheets model history"):
 *   1. In your Google Sheet: Extensions > Apps Script, paste this whole file, click Save.
 *   2. Choose installDailyTrigger in the function menu, click Run, and approve the permissions.
 *   3. Reload the sheet: a "Model" menu appears with "Update now".
 */
const BASE = 'https://carsonbillig.github.io/football-model/history/';
const HEADER_BG = '#2f5d4a';
const RESULT_COLUMNS = ['atsWin', 'winnerResult', 'totalResult', 'modelOnlyAtsWin'];

function onOpen() {
  SpreadsheetApp.getUi().createMenu('Model').addItem('Update now', 'syncModelHistory').addToUi();
}

function syncModelHistory() {
  const ss = SpreadsheetApp.getActive();
  const index = JSON.parse(fetchText_('index.json'));
  writeTab_(ss, 'Summary', parse_(fetchText_('summary.csv')), false);
  index.forEach(function (w) { writeTab_(ss, w.tab, parse_(fetchText_(w.file)), true); });

  // Summary first, then newest weeks first.
  const order = ['Summary'].concat(index.slice().reverse().map(function (w) { return w.tab; }));
  order.forEach(function (name, i) {
    const sh = ss.getSheetByName(name);
    if (sh) { ss.setActiveSheet(sh); ss.moveActiveSheet(i + 1); }
  });
  const blank = ss.getSheetByName('Sheet1');
  if (blank && blank.getLastRow() === 0 && ss.getSheets().length > 1) ss.deleteSheet(blank);
  ss.setActiveSheet(ss.getSheetByName('Summary'));
}

function installDailyTrigger() {
  ScriptApp.getProjectTriggers().forEach(function (t) {
    if (t.getHandlerFunction() === 'syncModelHistory') ScriptApp.deleteTrigger(t);
  });
  ScriptApp.newTrigger('syncModelHistory').timeBased().everyDays(1).atHour(9).create();
  syncModelHistory();
}

function fetchText_(file) {
  // the timestamp skips GitHub's cache so a fresh run shows up right away
  return UrlFetchApp.fetch(BASE + file + '?t=' + Date.now(), { muteHttpExceptions: false }).getContentText();
}

function parse_(csv) {
  return Utilities.parseCsv(csv).map(function (row) {
    return row.map(function (v) { return /^-?\d+(\.\d+)?$/.test(v) ? Number(v) : v; });
  });
}

function writeTab_(ss, name, rows, isWeek) {
  if (!rows.length) return;
  const sh = ss.getSheetByName(name) || ss.insertSheet(name);
  sh.clear();
  const width = rows[0].length;
  sh.getRange(1, 1, rows.length, width).setValues(rows);
  sh.getRange(1, 1, 1, width).setFontWeight('bold').setFontColor('#ffffff').setBackground(HEADER_BG);
  sh.setFrozenRows(1);

  if (isWeek) {
    // colour the result columns
    const header = rows[0];
    RESULT_COLUMNS.forEach(function (col) {
      const c = header.indexOf(col);
      if (c < 0 || rows.length < 2) return;
      const colours = rows.slice(1).map(function (r) {
        return [r[c] === 'Win' ? '#d9f2e1' : r[c] === 'Loss' ? '#f8dcd8' : r[c] === 'Push' ? '#eeeeee' : null];
      });
      sh.getRange(2, c + 1, colours.length, 1).setBackgrounds(colours);
    });
    // Wins / Losses / Pushes block to the right, like a classic tracking sheet
    const col = width + 2;
    const block = [['', 'Wins', 'Losses', 'Pushes']];
    [['Spread (ATS)', 'atsWin'], ['Winner', 'winnerResult'], ['Total', 'totalResult'], ['Model-only ATS', 'modelOnlyAtsWin']]
      .forEach(function (pair) {
        const c = header.indexOf(pair[1]);
        const letter = columnLetter_(c + 1);
        block.push([pair[0], '=COUNTIF(' + letter + ':' + letter + ',"Win")',
                    '=COUNTIF(' + letter + ':' + letter + ',"Loss")', '=COUNTIF(' + letter + ':' + letter + ',"Push")']);
      });
    sh.getRange(1, col, block.length, 4).setValues(block);
    sh.getRange(1, col, 1, 4).setFontWeight('bold').setFontColor('#ffffff').setBackground(HEADER_BG);
  }
  sh.autoResizeColumns(1, sh.getLastColumn());
}

function columnLetter_(n) {
  let s = '';
  while (n > 0) { const m = (n - 1) % 26; s = String.fromCharCode(65 + m) + s; n = Math.floor((n - 1) / 26); }
  return s;
}

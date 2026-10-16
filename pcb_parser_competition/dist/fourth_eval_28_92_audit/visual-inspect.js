await page.goto('http://127.0.0.1:49492/', {waitUntil:'domcontentloaded'});
return await tabbit.observe({frames:'all',maxChars:4500});

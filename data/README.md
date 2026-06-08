# Data README

Thu muc nay tach du lieu theo dung vai tro trong do an.

## `fragrantica/`

Du lieu catalog tu Fragrantica/Kaggle. Phan nay dung de lay metadata co cau truc nhu ten nuoc hoa, hang, note, accord, rating/votes va lam seed ban dau cho pipeline tim review tren Reddit.

## `reddit_crawled/`

Du lieu crawl Reddit va cac file phai sinh tu review da crawl. Day la pham vi chinh cho cac bai toan hoc may trong slide va bao cao: 636 nuoc hoa co review Reddit, 2.159 review records va 636 LLM profiles.

## `database/`

Ban SQLite cua pipeline sau khi nap catalog, crawl Reddit, rut trich profile bang DeepSeek va tinh similarity.

## `data_summary.json`

Tom tat nhanh so dong/mau cua cac bang chinh dung trong bao cao.

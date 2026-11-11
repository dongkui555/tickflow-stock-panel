from datetime import date, timedelta

import polars as pl
import pytest

from app.services import stockformer_entry_cost_research as research
from app.strategy import paper

EXECUTION=dict(initial_cash=10000.,max_positions=3,commission_pct=.0003,stamp_tax_pct=.0005,
    slippage_bps=5.,minimum_commission=5.)


def test_cost_waterfall_matches_exact_paper_cash_return_and_minimum_fee():
    r=research.cost_waterfall(100,10.,10.5,EXECUTION)
    buy=paper.apply_slippage(10.,'buy',5.)
    sell=paper.apply_slippage(10.5,'sell',5.)
    outlay=100*buy+paper.buy_fee(100,buy,.0003)
    net=(100*sell-paper.sell_fee(100,sell,.0003,.0005)-outlay)/outlay
    assert r['net_absolute']==pytest.approx(net)
    assert r['cost_total']==pytest.approx(.05-net)
    assert sum(r[k] for k in research.COST_PARTS)==pytest.approx(r['cost_total'])
    assert r['minimum_commission_drag']>r['slippage_drag']>0
    assert r['minimum_commission_extra_yuan']>9
    assert r['net_without_minimum']>r['net_absolute']


def test_large_notional_has_no_minimum_fee_increment_and_zero_rates_are_supported():
    r=research.cost_waterfall(10000,10.,10.5,EXECUTION)
    assert r['minimum_commission_drag']==pytest.approx(0.)
    assert r['minimum_commission_extra_yuan']==pytest.approx(0.)
    assert r['proportional_commission_drag']>0 and r['stamp_tax_drag']>0
    free={**EXECUTION,'commission_pct':0.,'stamp_tax_pct':0.,'slippage_bps':0.}
    r=research.cost_waterfall(100,10.,10.,free)
    assert r['slippage_drag']==r['proportional_commission_drag']==r['stamp_tax_drag']==0
    assert r['cost_total']==pytest.approx(10/1005)


@pytest.mark.parametrize('qty,entry,exit_price,changes',[
    (0,10.,11.,{}),(130,10.,11.,{}),(100,float('nan'),11.,{}),
    (100,10.,0.,{}),(100,10.,11.,{'commission_pct':-.1}),
    (100,10.,11.,{'slippage_bps':10000.}),(100,10.,11.,{'minimum_commission':0.}),
])
def test_invalid_cost_inputs_fail_closed(qty,entry,exit_price,changes):
    with pytest.raises(ValueError):
        research.cost_waterfall(qty,entry,exit_price,{**EXECUTION,**changes})


def frame(days=20):
    rows=[]
    for d in range(days):
        for i in range(4):
            net=.01*(i-1)
            rows.append(dict(symbol=str(i),date=date(2025,1,1)+timedelta(days=d),percentile=(i+.5)/4,
                pred_ret_raw_5d=float(i),_necessary_entry_pool=True,_entry_rules_pass=i!=3,
                _market_gate_pass=True,_cost_verified=True,market_gate='open',net_industry=net,
                gross_industry=net+.005,net_absolute=net+.02,gross_absolute=net+.025,cost_total=.005,
                slippage_drag=.001,proportional_commission_drag=.0006,stamp_tax_drag=.0005,
                minimum_commission_drag=.0029,net_without_minimum_industry=net+.0029,
                minimum_commission_extra_yuan=8.,cash_outlay=2000.))
    return pl.DataFrame(rows)


def test_full_reference_percentiles_are_not_reranked_after_entry_or_missing_label():
    f=frame()
    calendar=f['date'].unique().sort().to_list()
    daily=research.daily_groups(f,calendar,'full_entry')
    top=daily.filter(pl.col('group')=='top20')
    assert top['rows'].sum()==0 and top['net_industry'].null_count()==20
    missing=f.with_columns(pl.when(pl.col('symbol')=='3').then(False).otherwise(pl.col('_cost_verified')).alias('_cost_verified'))
    top=research.daily_groups(missing,calendar,'necessary').filter(pl.col('group')=='top20')
    assert top['rows'].sum()==20 and top['labeled_rows'].sum()==0
    assert top['net_industry'].null_count()==20
    with pytest.raises(ValueError,match='Duplicate'):
        research.daily_groups(pl.concat([f,f]),calendar,'necessary')


def test_equal_date_means_intervals_and_nonoverlap_offsets_preserve_calendar_gaps():
    f=frame()
    calendar=f['date'].unique().sort().to_list()
    daily=research.daily_groups(f,calendar,'necessary').filter(pl.col('group')=='top20')
    a=research.summarize(daily,calendar)
    b=research.summarize(daily,calendar)
    assert a==b and a['net_industry']['mean']==pytest.approx(.02)
    assert a['net_industry']['ci95']==pytest.approx([.02,.02])
    assert a['nonoverlapping_offsets'][0]['days']==4
    sparse=daily.filter(pl.col('date').is_in([calendar[0],calendar[2]]))
    s=research.summarize(sparse,calendar)
    assert s['net_industry']['ci95'] is None
    assert s['nonoverlapping_offsets'][0]['days']==1
    assert s['nonoverlapping_offsets'][1]['days']==0
    assert s['nonoverlapping_offsets'][2]['days']==1
    with pytest.raises(ValueError,match='calendar'):
        research.summarize(daily,calendar[1:])

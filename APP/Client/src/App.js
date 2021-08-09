import React, { Component } from 'react';
import styled from 'styled-components';
import './App.css';
import { withStyles, makeStyles } from '@material-ui/core/styles';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import Paper from '@material-ui/core/Paper';
import { yellow } from '@material-ui/core/colors';
import Tippy from '@tippy.js/react';


document.body.style.background = "#bfd6f6";
const BTN = styled.button`
  background-color: gray;
  display: inline-block;
  font-size: 1em;
  border: 2px solid whitesmoke;
  border-radius: 3px;
  display: block;
  margin-right: 100px;
  width:100px ;
  height: 40px;
  background-color: black;
  box-shadow: 40px;
  color: whitesmoke;

`;
const Title = styled.h1`
  font-size: 1.5em;
  text-align: right;
  padding-right:3em;
  color: white;
`;
const Wrapper = styled.section`
  background:   black;
  padding: 2em;
  box-shadow: 60px;
  padding-left: 10px;
  padding-right: 10px;
`;

const Input = styled.input`
  padding: 0.5em;
  margin: 0.5em;
  text-align: right;
  background: white ;
  border: none;
  border-radius: 3px;
  width: 1135px;
  height: 100px;
  font-weight  : bold;
`;

const insert = styled.text`
  text-align: right;
`;


const StringContent = () => (
    <Tippy content="Hello">
      <button>My button</button>
    </Tippy>
  );
   
  const JSXContent = () => (
    <Tippy content={<span>Tooltip</span>}>
      <button>My button</button>
    </Tippy>
  );

const StyledTableCell = withStyles((theme) => ({
    head: {
        backgroundColor: theme.palette.common.black,
        color: theme.palette.common.white,
    },
    body: {
        fontSize: 14,
    },
}))(TableCell);


const StyledTableRow = withStyles((theme) => ({
    root: {
        '&:nth-of-type(odd)': {
            backgroundColor: theme.palette.action.hover,
        },
    },
}))(TableRow);

const useStyles = makeStyles({
    table: {
        minWidth: 700,
    },
});

function entityBreaking(val){
	const entities = val.split('^');
	const length = entities.length;
	if(length == 1){
		if(val == 'O'){
			return val;
		}else{
			return entities[0].substring(2);
			}
	}else{
		var lastWord = entities[length-1];
		if(lastWord != 'O'){
			var singelton = lastWord.substring(2);
			return singelton;
		}else{
			return lastWord;
		}
	}
}

function fullEntity(val){
	if(val != 'O'){
		switch (val) {
			case 'ANG':
				return val+" (language)";
				break;
			case 'DUC':
				return val+" (product)";
				break;
			case 'EVE':
				return val+" (event)";
				break;
			case 'FAC':
				return val+" (facility)";
				break;
			case 'GPE':
				return val+" (geo-political entity)";
				break;
			case 'LOC':
				return val+" (location)";
				break;
			case 'ORG':
				return val+" (organization)";
				break;
			case 'PER':
				return val+" (person)";
				break;
			case 'WOA':
				return val+" (work-of-art)";
				break;
		}
	}else{
		return val;
	}
}



class App extends Component {
    state = {
        characters: [],
        valueInput: "",
        data: [],
        data2: [],
        colors: [],
        modeldata: [],
        flag: false,
        print: [],
        printScores: [],
        Num_of_entities: [],
        table :[]
    };

    removeCharacter = index => {
        const { characters } = this.state;

        this.setState({
            characters: characters.filter((character, i) => {
                return i !== index;
            })
        });
    }

    handleSubmit = character => {
        this.setState({ characters: [...this.state.characters, character] });
    }
    dorgetter = () => {
        let m = [...this.state.modeldata];
        let data = [...this.state.data];
        var words = []
        var colors2 = []
        var num_entities = 0;
        var sentesce = "";
		var score = 0 ;
        var print_scores = []
        var table_list = []
        for (let i = 0; i < data.length; i++) {
            sentesce += data[i]
        }
        words = sentesce.split(" ");
        for (let i = 0; i < m.length; i++) {
            var Type = entityBreaking(m[i].entity)
            var color = this.entity2color(Type)
            colors2[i] = color
        }

        


        this.setState({ colors: [...colors2] })
        this.setState({ data2: [...words] })
        let indents = [];
        for (var i = 0; i < this.state.data2.length; i++) {
            if (m[i] != null && entityBreaking(m[i].entity) != 'O') {
              //  console.log(this.state.colors[i]);
                num_entities += 1;
            }
        }
        this.setState({Num_of_entities: num_entities});
        indents.push(<li><h>
            <mark style={{background: yellow, padding: '0.45em 0.6em', lineHeight: '1', borderRadius: '0.35em' }}>Entities<span style={{ lineHeight: '1', borderRadius: '0.35em', verticalAlign: 'middle', marginLeft: '0.5rem', fontSize: '0.8em', fontWeight: 'bold' }}>Num : {num_entities}</span></mark>
         </h></li>);
        print_scores.push(
            <h1>
                Probabilities:
            </h1>);
        for (var i = 0; i < this.state.data2.length; i++) {
            if (m[i] != null && entityBreaking(m[i].entity) != 'O') {
                console.log(this.state.colors[i]);
                num_entities += 1;
				score = m[i].score * 100;
                print_scores.push(<ul><mark style={{ background: this.state.colors[i], padding: '0.45em 0.6em', margin: '0 0.25em', lineHeight: '1', borderRadius: '0.35em'}}>
                    {entityBreaking(this.state.data2[i]) + "\t" + score.toFixed(3) + "\n"}
                    <span style={{ lineHeight: '1', borderRadius: '0.35em', verticalAlign: 'middle', marginLeft: '0.5rem', fontSize: '0.8em', fontWeight: 'bold'}} key>{entityBreaking(m[i].entity)}</span>
                </mark></ul>);
                indents.push(<Tippy content = {score.toFixed(3)} ><mark style={{ background: this.state.colors[i], padding: '0.45em 0.6em', margin: '0 0.25em', lineHeight: '1', borderRadius: '0.35em' }}>
                    {this.state.data2[i]}
                </mark></Tippy> );
            }
            else indents.push(this.state.data2[i] + " ");
        }

        indents.push(<li><h>
            <mark style={{background: yellow, padding: '0.45em 0.6em', lineHeight: '1', borderRadius: '0.35em' }}>LEGEND<span style={{ lineHeight: '1', borderRadius: '0.35em', verticalAlign: 'middle', marginLeft: '0.5rem', fontSize: '0.8em', fontWeight: 'bold' }}></span></mark>
         </h></li>);
        let list_dup = []

        for (var i = 0; i < m.length; i++) {
            if (m[i] != null && entityBreaking(m[i].entity) != 'O') {
                if (!list_dup.includes(fullEntity(entityBreaking(m[i].entity)))){
                list_dup.push(fullEntity(entityBreaking(m[i].entity)))
                indents.push(<mark style={{ background: this.state.colors[i], padding: '0.45em 0.6em', margin: '0 0.25em', lineHeight: '1', borderRadius: '0.35em'}}>
                    {fullEntity(entityBreaking(m[i].entity))}
                    <span style={{ lineHeight: '1', borderRadius: '0.35em', verticalAlign: 'middle', marginLeft: '0.5rem', fontSize: '0.8em', fontWeight: 'bold'}} key></span>
                </mark>);
            }
        }
        }

        this.setState({ print: [...indents] });
        this.setState({ printScores: [...print_scores] });
        this.create_table();
    }

    create_table(){
        var table2 = []
        table2.push(<TableContainer component={Paper}  align= "justify" padding ="2em">
            <Table  aria-label="customized table" fontWeight = "bold">
                <TableHead  backgroundColor ="blue">
                    <TableRow>
                        <StyledTableCell style = {{align :"justify",fontWeight :"bold"}} >Words</StyledTableCell>
                        <StyledTableCell style = {{align :"justify",fontWeight :"bold"}}> Probabilities</StyledTableCell>
                        <StyledTableCell style = {{align :"justify",fontWeight :"bold"}}>Entity</StyledTableCell>
                    </TableRow>
                </TableHead>
            {this.state.modeldata.map((row,i) =>(
                 <StyledTableRow key={row.word}>
                 <StyledTableCell style = {{align :"justify",fontWeight :"bold" , backgroundColor : this.state.colors[i]}} component="th" scope="row">
                   {row.word}
                 </StyledTableCell>
                 <StyledTableCell style = {{align :"justify" ,fontWeight :"bold" , backgroundColor : this.state.colors[i]}}>{(row.score * 100).toFixed(3)}</StyledTableCell>
                 <StyledTableCell style = {{align :"justify" ,fontWeight :"bold" , backgroundColor : this.state.colors[i]}}>{entityBreaking(row.entity)}</StyledTableCell>
               </StyledTableRow>
             ))}
            </Table>
        </TableContainer>);
        this.setState({ table: [...table2] });

    }
    
    entity2color(val) {
        switch (val) {
            case 'ANG':
                return "orange";
                break;
			case 'DUC':
                return "red";
                break;
			case 'EVE':
                return "blue";
                break;
			case 'FAC':
                return "brown";
                break;
			case 'GPE':
                return "purple";
                break;
			case 'LOC':
                return "yellow";
                break;
			case 'ORG':
                return "green";
                break;
			case 'PER':
                return "#f1b507";
                break;
			case 'WOA':
                return "#07bcf1";
                break;
            case 'O':
			case 'PAD':
                return "";
                break;
            default:
                return 'white';
        }
    }
    sendHttp = () => {
        const requestOptions = {
            method: 'POST',
            mode: 'no-cors',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: this.state.valueInput })
        };

        fetch('http://127.0.0.1:105/new_code', requestOptions)
            .then(response => response.text())
    }
    getData = () => {
        var text = require('./RawText.json'); //(with path)
        var jsontext = require('./textJSON.json'); //(with path)
        this.setState({ data: text })
        this.setState({ modeldata: jsontext })
        this.dorgetter()

    }
    handleChange = (c) => {
        this.setState({ valueInput: c.target.value })
    }

    render() {
        const { print } = this.state;
        const {table} = this.state;
        
        return (
            <div className="App">  
                <Wrapper><img src={require('./Bert2.png')} align = "0.5em" height="120em"   justifyContent = 'right'   align = 'right'/> 

                    <Title> 
                        Welcome to NER with
                        <mark style={{ background: "#cbdadb", padding: '0.45em 0.6em', margin: '0 0.25em', lineHeight: '1', borderRadius: '0.35em' }}>
                            BERT
                            <span style={{ lineHeight: '1', borderRadius: '0.35em', verticalAlign: 'middle', marginLeft: '0.5rem', fontSize: '0.8em', fontWeight: 'bold' }}> MODEL</span>
                        </mark> 
                    </Title>
                </Wrapper>
                
                <Input type="text" id="fname" name="fname" value={this.state.valueInput} onChange={(e) => { this.handleChange(e) }} align = 'right' />
                <BTN onClick={this.sendHttp}>הגשה</BTN>
                <BTN onClick={this.getData} type="submit">תוצאה</BTN>
                
                <div>  {print}   </div>
             
                <div style= {{ display :"block", padding : "1.5em"}}>  {table}   </div>


               
                </div>
        );
    }
}
export default App;